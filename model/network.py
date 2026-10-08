import math
import os
import timm
import torch
import logging
import torchvision
import torch.nn.functional as F
from glob import glob
from torch import nn
import torchvision
from os.path import join
from transformers import AutoModel, ViTModel
from googledrivedownloader import download_file_from_google_drive

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock
from model.functional import ReverseLayerF
from model.pix2pix_networks.networks import UnetGenerator, GANLoss, NLayerDiscriminator, get_scheduler
from model.sync_batchnorm import convert_model


PRETRAINED_MODELS = {
    "resnet18_places": "1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5",
    "resnet50_places": "1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC",
    "resnet101_places": "1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa",
    "vgg16_places": "1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu",
    "resnet18_gldv2": "1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke",
    "resnet50_gldv2": "1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70",
    "resnet101_gldv2": "1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE",
    "vgg16_gldv2": "10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o",
}


def get_pretrained_model(args):
    if args.pretrain == "places":
        num_classes = 365
    elif args.pretrain == "gldv2":
        num_classes = 512
    else:
        raise NotImplementedError(f"Unsupported ResNet pretrain: {args.pretrain}")

    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    else:
        raise NotImplementedError(f"Unsupported backbone for pretrain: {args.backbone}")

    model_name = args.backbone.split("conv")[0] + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name + ".pth")
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logging.info(f"Downloading {model_name} checkpoint to {file_path}")
        download_file_from_google_drive(
            file_id=PRETRAINED_MODELS[model_name],
            dest_path=file_path,
        )
        if (
            not os.path.exists(file_path)
            or os.path.getsize(file_path) < 1024 * 1024
            or _looks_like_html_error(file_path)
        ):
            if os.path.exists(file_path):
                os.remove(file_path)
            raise RuntimeError(
                f"Failed to download a valid {model_name} checkpoint. "
                f"The configured Google Drive id may be expired. "
                f"Place a valid checkpoint at {file_path}, or use --pretrain imagenet."
            )

    logging.info(f"Loading {model_name} checkpoint from {file_path}")
    if os.path.getsize(file_path) < 1024 * 1024 or _looks_like_html_error(file_path):
        raise RuntimeError(
            f"Invalid checkpoint file at {file_path}. "
            f"It looks like an HTML/error page or an incomplete download. "
            f"Delete it and place a valid {model_name} checkpoint there."
        )
    state_dict = torch.load(file_path, map_location=torch.device("cpu"), weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        updated_state_dict = {}
        for key, value in state_dict.items():
            updated_state_dict[key.replace("module.encoder.", "")] = value
        updated_state_dict.pop("fc.weight", None)
        updated_state_dict.pop("fc.bias", None)
        state_dict = updated_state_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info(
        f"Loaded {model_name} checkpoint | missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
    )
    return model


def _looks_like_html_error(file_path):
    with open(file_path, "rb") as f:
        prefix = f.read(256).lstrip().lower()
    return prefix.startswith(b"<html") or prefix.startswith(b"<!doctype html")


def _strip_checkpoint_prefix(state_dict):
    clean = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        clean[key] = value
    return clean


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return _strip_checkpoint_prefix(ckpt[key])
        if all(isinstance(value, torch.Tensor) for value in ckpt.values()):
            return _strip_checkpoint_prefix(ckpt)
    raise ValueError("Unsupported checkpoint format for DINOv3 ConvNeXt checkpoint.")


def _resolve_local_checkpoint_path(checkpoint_source):
    if os.path.isfile(checkpoint_source):
        return checkpoint_source
    if os.path.isdir(checkpoint_source):
        candidates = []
        for pattern in ["*.pth", "*.pt", "*.bin"]:
            candidates.extend(glob(join(checkpoint_source, pattern)))
        candidates = sorted(candidates)
        if len(candidates) == 0:
            return None
        if len(candidates) > 1:
            logging.warning(
                f"Found multiple checkpoint files under {checkpoint_source}, using the first one: {candidates[0]}"
            )
        return candidates[0]
    return None


def _convert_convnext_ckpt_to_timm(state_dict):
    converted = {}
    dropped = []
    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("downsample_layers.0."):
            new_key = new_key.replace("downsample_layers.0.", "stem.", 1)
        elif new_key.startswith("downsample_layers.1."):
            new_key = new_key.replace("downsample_layers.1.", "stages.1.downsample.", 1)
        elif new_key.startswith("downsample_layers.2."):
            new_key = new_key.replace("downsample_layers.2.", "stages.2.downsample.", 1)
        elif new_key.startswith("downsample_layers.3."):
            new_key = new_key.replace("downsample_layers.3.", "stages.3.downsample.", 1)

        new_key = new_key.replace(".dwconv.", ".conv_dw.")
        new_key = new_key.replace(".pwconv1.", ".mlp.fc1.")
        new_key = new_key.replace(".pwconv2.", ".mlp.fc2.")

        if new_key.startswith("stages."):
            parts = new_key.split(".")
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                new_key = f"stages.{parts[1]}.blocks.{parts[2]}." + ".".join(parts[3:])

        if new_key == "norm.weight":
            new_key = "head.norm.weight"
        elif new_key == "norm.bias":
            new_key = "head.norm.bias"
        elif new_key.startswith("norms."):
            dropped.append(key)
            continue

        converted[new_key] = value
    return converted, dropped


def _convert_dinov3_vit_ckpt_to_timm(state_dict):
    converted = {}
    dropped = []
    for key, value in state_dict.items():
        new_key = key

        if new_key == "storage_tokens":
            new_key = "reg_token"

        new_key = new_key.replace(".ls1.gamma", ".gamma_1")
        new_key = new_key.replace(".ls2.gamma", ".gamma_2")

        if (
            new_key == "mask_token"
            or new_key == "rope_embed.periods"
            or new_key.startswith("local_cls_norm.")
            or new_key.endswith(".attn.qkv.bias_mask")
            or new_key.endswith(".attn.qkv.bias")
        ):
            dropped.append(key)
            continue

        converted[new_key] = value
    return converted, dropped


class DINOv3ConvNeXtBackbone(nn.Module):
    def __init__(self, checkpoint_source, timm_model_name, freeze_stages=0):
        super().__init__()
        ckpt_path = _resolve_local_checkpoint_path(checkpoint_source)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Could not find a DINOv3 ConvNeXt checkpoint under: {checkpoint_source}"
            )

        logging.info(f"Loading local timm DINOv3 ConvNeXt checkpoint: {ckpt_path}")
        logging.info(f"Using timm architecture: {timm_model_name}")
        backbone = timm.create_model(timm_model_name, pretrained=False)
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(raw_ckpt)
        state_dict, dropped = _convert_convnext_ckpt_to_timm(state_dict)
        if dropped:
            logging.info(f"Dropped {len(dropped)} incompatible ConvNeXt checkpoint keys")
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded local ConvNeXt checkpoint | missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )
        logging.info(f"Sample missing keys: {missing[:5] if missing else []}")
        logging.info(f"Sample unexpected keys: {unexpected[:5] if unexpected else []}")

        self.stem = backbone.stem
        self.stages = backbone.stages
        self.base_output_dim = int(getattr(backbone, "num_features", 768))
        self._freeze_stages(freeze_stages)

    def _freeze_stages(self, freeze_stages):
        if freeze_stages <= 0:
            logging.info("DINOv3 ConvNeXt backbone is fully trainable.")
            return

        freeze_stages = min(max(int(freeze_stages), 0), 4)
        modules_to_freeze = [self.stem]
        for stage_idx in range(min(freeze_stages, len(self.stages))):
            modules_to_freeze.append(self.stages[stage_idx])

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        logging.info(f"Freeze ConvNeXt stem and the first {freeze_stages} stage(s).")

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x


class DINOv3ViTBackbone(nn.Module):
    def __init__(self, model_source):
        super().__init__()
        logging.info(f"Loading DINOv3 ViT backbone: {model_source}")
        ckpt_path = _resolve_local_checkpoint_path(model_source)
        if ckpt_path is not None:
            timm_model_name = "vit_small_patch16_dinov3.lvd1689m"
            logging.info(f"Loading local timm DINOv3 ViT-S checkpoint: {ckpt_path}")
            logging.info(f"Using timm architecture: {timm_model_name}")
            self.backbone = timm.create_model(
                timm_model_name,
                pretrained=False,
                num_classes=0,
            )
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = _extract_state_dict(raw_ckpt)
            state_dict, dropped = _convert_dinov3_vit_ckpt_to_timm(state_dict)
            if dropped:
                logging.info(f"Dropped {len(dropped)} incompatible DINOv3 ViT checkpoint keys")
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            logging.info(
                f"Loaded local DINOv3 ViT checkpoint | missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
            )
            logging.info(f"Sample missing keys: {missing[:5] if missing else []}")
            logging.info(f"Sample unexpected keys: {unexpected[:5] if unexpected else []}")
            self.backend = "timm"
            self.base_output_dim = int(getattr(self.backbone, "num_features", getattr(self.backbone, "embed_dim", 384)))
            self.num_register_tokens = max(int(getattr(self.backbone, "num_prefix_tokens", 1)) - 1, 0)
            patch_size = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", 16)
            if isinstance(patch_size, (tuple, list)):
                patch_size = patch_size[0]
            self.patch_size = int(patch_size)
        else:
            self.backbone = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
            )
            self.backend = "hf"
            self.base_output_dim = int(getattr(self.backbone.config, "hidden_size", 384))
            self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 4))
            self.patch_size = 16

    def _tokens_to_map(self, tokens, x, strip_prefix=True):
        if tokens.ndim != 3:
            return tokens
        if strip_prefix:
            tokens = tokens[:, 1 + self.num_register_tokens :, :]
        b, n, c = tokens.shape
        side = int(math.isqrt(n))
        if side * side != n:
            h = x.shape[-2] // self.patch_size
            w = x.shape[-1] // self.patch_size
            if h * w != n:
                raise RuntimeError(
                    f"Cannot reshape ViT tokens to map: tokens={n}, patch_size={self.patch_size}, input={tuple(x.shape)}"
                )
            side_h, side_w = h, w
        else:
            side_h, side_w = side, side
        return tokens.transpose(1, 2).contiguous().view(b, c, side_h, side_w)

    def forward(self, x):
        if self.backend == "timm":
            if hasattr(self.backbone, "forward_features"):
                feat = self.backbone.forward_features(x)
            else:
                feat = self.backbone(x)
            if isinstance(feat, dict):
                if "x_norm_patchtokens" in feat:
                    return self._tokens_to_map(feat["x_norm_patchtokens"], x, strip_prefix=False)
                if "x_prenorm" in feat:
                    feat = feat["x_prenorm"]
                elif "x_norm" in feat:
                    feat = feat["x_norm"]
                else:
                    for value in feat.values():
                        if torch.is_tensor(value):
                            feat = value
                        break
            if torch.is_tensor(feat):
                return self._tokens_to_map(feat, x)
            return feat

        feat = self.backbone(x)
        if hasattr(feat, "last_hidden_state"):
            return self._tokens_to_map(feat.last_hidden_state, x)
        if isinstance(feat, (list, tuple)):
            return feat[0]
        return feat


class ModalityAwareAffineRemap(nn.Module):
    """Modality-aware affine remap with Gaussian perturbation during training.

    modality:
        - 0 / "rgb": satellite / database branch
        - 1 / "ir" : thermal / query branch
    """

    def __init__(self, channels, sigma_gamma=0.02, sigma_beta=0.01):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.mu_gamma_rgb = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.mu_beta_rgb = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.mu_gamma_ir = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.mu_beta_ir = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sigma_gamma = float(sigma_gamma)
        self.sigma_beta = float(sigma_beta)

    def _sample_affine(self, gamma_mu, beta_mu):
        if not self.training:
            return gamma_mu, beta_mu

        if self.sigma_gamma > 0:
            gamma = gamma_mu + torch.randn_like(gamma_mu) * self.sigma_gamma
        else:
            gamma = gamma_mu
        if self.sigma_beta > 0:
            beta = beta_mu + torch.randn_like(beta_mu) * self.sigma_beta
        else:
            beta = beta_mu
        return gamma, beta

    def forward(self, x, modality=None):
        if modality is None:
            return x

        x = self.norm(x)

        if isinstance(modality, str):
            if modality == "rgb":
                gamma, beta = self._sample_affine(self.mu_gamma_rgb, self.mu_beta_rgb)
                return gamma * x + beta
            if modality == "ir":
                gamma, beta = self._sample_affine(self.mu_gamma_ir, self.mu_beta_ir)
                return gamma * x + beta
            raise ValueError(f"Unsupported modality string: {modality}")

        if not torch.is_tensor(modality):
            raise TypeError("modality must be None, a string, or a tensor.")

        modality = modality.to(device=x.device)
        if modality.ndim != 1 or modality.shape[0] != x.shape[0]:
            raise ValueError(
                f"Modality tensor must have shape [B]. Got {tuple(modality.shape)} for batch {x.shape[0]}."
            )

        modality = modality.view(-1, 1, 1, 1).float()
        gamma_rgb, beta_rgb = self._sample_affine(self.mu_gamma_rgb, self.mu_beta_rgb)
        gamma_ir, beta_ir = self._sample_affine(self.mu_gamma_ir, self.mu_beta_ir)
        gamma = gamma_rgb * (1.0 - modality) + gamma_ir * modality
        beta = beta_rgb * (1.0 - modality) + beta_ir * modality
        return gamma * x + beta


class DirectionalStructureModule(nn.Module):
    """Enhance directional and low-frequency structure before modality alignment."""

    def __init__(
        self,
        channels,
        num_dirs=8,
        low_kernel=5,
        use_semantic_branch=True,
        use_directional_branch=True,
        use_low_frequency_branch=True,
    ):
        super().__init__()
        if num_dirs not in (4, 8, 12):
            raise ValueError(f"DSM supports 4, 8, or 12 directions, got {num_dirs}")
        if not any([use_semantic_branch, use_directional_branch, use_low_frequency_branch]):
            raise ValueError("At least one DSM branch must be enabled.")

        self.num_dirs = num_dirs
        self.use_semantic_branch = bool(use_semantic_branch)
        self.use_directional_branch = bool(use_directional_branch)
        self.use_low_frequency_branch = bool(use_low_frequency_branch)
        hidden = max(channels // 4, 32)

        self.sem_branch = nn.Sequential(#定义一个顺序前向的神经网络模块，数据会一层一层按顺序过
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),#1x1 conv：通道混合，重新组织语义信息
            nn.GroupNorm(1, channels),#归一化
            nn.SiLU(inplace=True),#激活函数：SiLU（Swish）平滑非线性
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),#3×3 深度可分离卷积，保留空间局部结构
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),#再次通道融合，重新整合通道信息
        )

        self.dir_branches = nn.ModuleList(#创建 num_dirs 个独立的 3×3 深度卷积层，组成方向分支，用于学习不同方向 / 不同频率 / 不同区域的空间特征。
            [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
                for _ in range(num_dirs)#8 个方向分支，每个分支都是3x3 depthwise conv
            ]
        )

        self.low_branch = nn.Sequential(#低频结构分支
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),#3×3 平均池化，平滑、去噪、低频提取
            nn.Conv2d(#用更大的感受野提取低频、大尺度结构
                channels,
                channels,
                kernel_size=low_kernel,#low_kernel=5
                padding=low_kernel // 2,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),#整合通道
        )

        self.fusion_mlp = nn.Sequential(#学习自适应权重（三个分支的权重），用来融合
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )

        self.post_norm = nn.GroupNorm(1, channels)#最终输出前做一次归一化，让特征分布更稳定
        self._init_directional_kernels()

    def _kirsch_templates(self):
        templates = [
            [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
            [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
            [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
            [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
            [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
            [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
            [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
            [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
        ]
        return [torch.tensor(t, dtype=torch.float32) / 15.0 for t in templates]

    def _generalized_compass_templates(self):
        """Create evenly spaced 3x3 directional templates for >8 branches.

        4 and 8 directions preserve the original Kirsch-style initialization exactly.
        For 12 directions, we keep the same "compass prior" idea but generate
        denser angular templates on the 3x3 grid so that each branch starts with
        a distinct orientation bias.
        """
        coords = torch.tensor(
            [
                [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0],
                [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
            ],
            dtype=torch.float32,
        )
        templates = []
        for k in range(self.num_dirs):
            theta = 2.0 * math.pi * k / float(self.num_dirs)
            direction = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float32)
            projection = coords @ direction
            kernel = projection.view(3, 3)
            kernel[1, 1] = 0.0
            pos = torch.clamp(kernel, min=0.0)
            neg = torch.clamp(kernel, max=0.0)
            kernel = 5.0 * pos + 3.0 * neg
            kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
            templates.append(kernel)
        return templates

    def _init_directional_kernels(self):#初始化方向卷积核
        templates = self._kirsch_templates()#获取 8 个 Kirsch 核
        if self.num_dirs == 4:#如果只需 4 方向，就取 上下左右 4 个核
            templates = [templates[i] for i in (0, 2, 4, 6)]
        elif self.num_dirs == 12:
            templates = self._generalized_compass_templates()

        for conv, kernel in zip(self.dir_branches, templates):#遍历所有方向分支卷积层
            with torch.no_grad():#不计算梯度（手动赋值）
                conv.weight.copy_(kernel.view(1, 1, 3, 3).repeat(conv.weight.shape[0], 1, 1, 1))#把 Kirsch 核直接赋值给卷积层权重，让每个方向分支变成 固定的边缘检测器

    def forward(self, x):
        f_sem = self.sem_branch(x) if self.use_semantic_branch else None#原始语义

        #方向响应
        if self.use_directional_branch:
            dir_responses = [branch(x) for branch in self.dir_branches]#让 x 依次经过每一个方向卷积层，得到K（k=8）个输出
            dir_stack = torch.stack(dir_responses, dim=1)  # B, K, C, H, W，把 K 个方向特征在 dim=1 堆叠
            dir_logits = dir_stack.mean(dim=2)  # B, K, H, W，对通道维度 C 求平均，得到每个方向、每个位置的响应强度
            dir_weights = torch.softmax(dir_logits, dim=1)#在方向维度K做softmax，得到每个位置 K 个方向的权重
            f_ori = (dir_stack * dir_weights.unsqueeze(2)).sum(dim=1)#方向加权融合，用学到的权重给 K 个方向特征加权求和，得到最终多方向边缘增强特征 f_ori
            dir_hist = dir_weights.mean(dim=(2, 3))#对 H、W 求平均，得到每张图的方向权重分布
            dir_logits_pooled = dir_logits.mean(dim=(2, 3))
        else:
            f_ori = None
            dir_hist = None
            dir_logits_pooled = None

        f_low = self.low_branch(x) if self.use_low_frequency_branch else None#低频结构

        fusion_logits = self.fusion_mlp(F.adaptive_avg_pool2d(x, output_size=1))#先对 x 全局池化 → (B, C, 1, 1)；送入 fusion_mlp，输出 3 个融合权重 logits，形状：(B, 3, 1, 1)
        fusion_weights = torch.softmax(fusion_logits, dim=1)#对 3 个 logits 做 softmax，得到 3 个归一化权重

        # Keep the original bitwise training path when all three branches are enabled.
        # Only use branch re-normalization for the actual ablation cases.
        if self.use_semantic_branch and self.use_directional_branch and self.use_low_frequency_branch:
            fused = (
                fusion_weights[:, 0:1] * f_sem
                + fusion_weights[:, 1:2] * f_ori
                + fusion_weights[:, 2:3] * f_low
            )
        else:
            active_weights = []
            active_features = []
            if self.use_semantic_branch:
                active_weights.append(fusion_weights[:, 0:1])
                active_features.append(f_sem)
            if self.use_directional_branch:
                active_weights.append(fusion_weights[:, 1:2])
                active_features.append(f_ori)
            if self.use_low_frequency_branch:
                active_weights.append(fusion_weights[:, 2:3])
                active_features.append(f_low)

            normalized_weights = torch.cat(active_weights, dim=1)
            normalized_weights = normalized_weights / normalized_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

            fused = 0.0
            for idx, feat_i in enumerate(active_features):
                fused = fused + normalized_weights[:, idx:idx + 1] * feat_i
        aux = {
            "dir_hist": dir_hist,
            "dir_logits_pooled": dir_logits_pooled,
        }
        return self.post_norm(x + fused), aux#残差连接：x + fused（原始特征 + 增强特征）


#一个图像检索/地理定位模型的骨架
#输入一张图 x (B,3,H,W) → 经过 backbone 提取特征图 feat_map (B,C,h,w) → 经过 aggregation（汇聚层） 把特征图变成一个全局向量 embedding emb (B,D)，用于检索/匹配
class GeoLocalizationNet(nn.Module):
    """所使用的网络由主干网和汇聚层组成"""

    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)#构建主干网络（如 ResNet、ViT、Swin 等），由 get_backbone(args) 函数返回
        self.arch_name = args.backbone
        self.self_att = False
        self.DA = args.DA
        self.directional_structure_module = None
        self.modality_affine_remap = None

        # 对 gem/spoc/mac/rmac 等汇聚方式，根据 l2 参数决定 L2 归一化的位置(暂时这一段都用不上）
        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:#最后都用 Flatten() 展平成向量
            if args.l2 == "before_pool":#先归一化，再池化
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":#先池化，再归一化
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":#不归一化
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        # 把 embedding 再映射到新维度
        if args.fc_output_dim != None:
            # 在汇聚层后加一个全连接层，调整特征维度
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim#更新 args.features_dim 为新维度

        if args.use_dsm:
            self.directional_structure_module = DirectionalStructureModule(
                args.features_dim,
                num_dirs=args.dsm_num_dirs,
                low_kernel=args.dsm_low_kernel,
                use_semantic_branch=args.dsm_use_semantic_branch,
                use_directional_branch=args.dsm_use_directional_branch,
                use_low_frequency_branch=args.dsm_use_low_frequency_branch,
            )

        if args.use_modality_affine_remap:
            self.modality_affine_remap = ModalityAwareAffineRemap(
                args.features_dim,
                sigma_gamma=args.modality_affine_sigma_gamma,
                sigma_beta=args.modality_affine_sigma_beta,
            )

        # NetVLAD/CRN 的 conv_layer：调整特征图通道数
        if args.aggregation in ["netvlad", "crn"] and args.conv_output_dim != None:
            # 对 NetVLAD/CRN 汇聚方式，增加一个 1×1 卷积层，用于降维
            actual_conv_output_dim = int(args.conv_output_dim / args.netvlad_clusters)#计算实际卷积输出维度 = 总维度 ÷ 聚类数
            logging.debug(f"Last conv layer dim: {actual_conv_output_dim}")#打印调试信息，记录卷积层维度
            if args.work_with_tokens:
                self.conv_layer = nn.Linear(args.features_dim, actual_conv_output_dim, bias=not args.add_bn)
            elif args.add_bn:#可选批量归一化，稳定训练（给swin用的）
                self.conv_layer = nn.Sequential(nn.Conv2d(args.features_dim, actual_conv_output_dim, 1, bias=False),
                                                nn.BatchNorm2d(actual_conv_output_dim), )
            else:
                self.conv_layer = nn.Conv2d(args.features_dim, actual_conv_output_dim, 1)
            args.features_dim = actual_conv_output_dim#更新特征维度为卷积输出维度

        self.aggregation = get_aggregation(args)

        # 如果开启 non_local，则添加 NonLocal 自注意力模块：，打破传统卷积仅能利用局部邻域信息的限制，让模型在提取特征时能关联到整个特征图的全局信息
        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                            channel_inner=args.channel_bottleneck)] * args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True

    def create_domain_classifier(self, args):
        if self.DA.startswith("DANN_before"):
            if self.DA == "DANN_before":
                return nn.Sequential(
                    nn.Linear(args.features_dim * 32 * 32, 1000, bias=False),
                    nn.BatchNorm1d(1000),
                    nn.ReLU(True),
                    nn.Linear(1000, 2),
                    nn.LogSoftmax(dim=1),
                )
            if self.DA == "DANN_before_conv":
                return nn.Sequential(
                    nn.Conv2d(args.features_dim, args.features_dim * 2, kernel_size=4, stride=2, bias=False),
                    nn.BatchNorm2d(args.features_dim * 2),
                    nn.ReLU(True),
                    nn.Conv2d(args.features_dim * 2, args.features_dim * 4, kernel_size=4, stride=2, bias=False),
                    nn.BatchNorm2d(args.features_dim * 4),
                    nn.ReLU(True),
                    nn.Conv2d(args.features_dim * 4, args.features_dim * 8, kernel_size=4, stride=2, bias=False),
                    nn.BatchNorm2d(args.features_dim * 8),
                    nn.ReLU(True),
                    nn.Conv2d(args.features_dim * 8, 2, kernel_size=2),
                    nn.Flatten(),
                    nn.LogSoftmax(dim=1),
                )
        if self.DA == "DANN_after":
            return nn.Sequential(
                nn.Linear(args.conv_output_dim, 100, bias=False),
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Linear(100, 2),
                nn.LogSoftmax(dim=1),
            )
        raise NotImplementedError(f"Unsupported domain adaptation mode: {self.DA}")


    def forward(self, x, is_train=False, alpha=0.0, return_domain=False, modality=None, return_aux=False):
        # 输入图像 x: (B,3,H,W)，主干网提取特征
        feat = self.backbone(x)
        dir_hist = None

        # ✅ timm Swin features_only=True 返回 list/tuple
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        elif hasattr(feat, "last_hidden_state"):
            feat = feat.last_hidden_state

        # ✅ NHWC -> NCHW（兼容少数 backbone 输出）
        if feat.ndim == 4:
            if feat.shape[1] not in (64, 128, 256, 512, 768) and feat.shape[-1] in (64, 128, 256, 512, 768):
                feat = feat.permute(0, 3, 1, 2).contiguous()

        if self.directional_structure_module is not None:
            feat, dsm_aux = self.directional_structure_module(feat)
            dir_hist = dsm_aux["dir_hist"]
            dir_logits_pooled = dsm_aux["dir_logits_pooled"]
        else:
            dir_logits_pooled = None

        if self.modality_affine_remap is not None:
            feat = self.modality_affine_remap(feat, modality=modality)

        if hasattr(self, "conv_layer") and self.conv_layer is not None:
            feat = self.conv_layer(feat)

        emb = self.aggregation(feat)
        reverse_feat = None
        if is_train and return_domain and self.DA is not None and self.DA != "none":
            if self.DA == "DANN_before":
                reverse_feat = ReverseLayerF.apply(feat.view(feat.shape[0], -1), alpha)
            elif self.DA == "DANN_before_conv":
                reverse_feat = ReverseLayerF.apply(feat, alpha)
            elif self.DA == "DANN_after":
                reverse_feat = ReverseLayerF.apply(emb, alpha)
            else:
                raise NotImplementedError(f"Unsupported domain adaptation mode: {self.DA}")
        if return_aux:
            aux = {
                "dir_hist": dir_hist,
                "dir_logits_pooled": dir_logits_pooled,
            }
            if return_domain:
                return emb, reverse_feat, aux
            return emb, aux
        if return_domain:
            return emb, reverse_feat
        return emb

    # @torch.no_grad()
    # def extract_layer2(self, x, normalize=True):
    #     """
    #     只用于 rerank：提取 ResNet 的 layer2 输出并池化成一个向量。
    #     - 不参与训练（no_grad）
    #     - 不改你现有 forward() 的主干行为
    #     返回：B x C 的向量（默认 L2 normalize）
    #     """
    #     # backbone 是 nn.Sequential(*list(resnet.children())[:-3 or :-2])
    #     # 对 resnet 来说 children 顺序通常是：
    #     # conv1, bn1, relu, maxpool, layer1, layer2, layer3, (layer4)
    #     modules = list(self.backbone.children())
    #     if len(modules) < 6:
    #         raise RuntimeError(
    #             f"Backbone seems not ResNet-like Sequential, children={len(modules)}. "
    #             "extract_layer2() currently supports ResNet-style backbone."
    #         )
    #
    #     # forward 到 layer2（包含 layer2）
    #     y = x
    #     for i, m in enumerate(modules):
    #         y = m(y)
    #         if i == 5:  # 0..5 -> 到 layer2 结束
    #             break
    #
    #     # 池化成向量：B x C
    #     y = torch.nn.functional.adaptive_avg_pool2d(y, (1, 1)).flatten(1)
    #
    #     if normalize:
    #         y = torch.nn.functional.normalize(y, p=2, dim=1)
    #     return y

#args.aggregation 是一个“字符串配置项”，用来指定：backbone 输出的特征图 (B,C,H,W) 要“用什么方法”变成全局向量 (B,D)
def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(
            clusters_num=args.netvlad_clusters,# 设置聚类中心的数量（如 16, 32, 64）
            dim=args.features_dim,  # 设置输入特征的通道维度
            work_with_tokens=args.work_with_tokens
        )


def get_backbone(args):
    # 根据架构的类型（CNN vs Transformer），聚合层的工作方式有所不同
    # args.work_with_tokens：设置一个标志位，指示后续的聚合层（如 GeM、NetVLAD）是否需要处理序列格式（Token）的特征
    # 如果 backbone 是 cct 或 vit（Transformer 架构），则设为 True；否则（如 ResNet）为 False
    args.work_with_tokens = (
        args.backbone.startswith('cct')
        or args.backbone.startswith('vit')
    )
    if args.backbone in ["dinov3convnextt", "dinov3convnextb"]:
        timm_model_name = {
            "dinov3convnextt": "convnext_tiny",
            "dinov3convnextb": "convnext_base",
        }[args.backbone]
        backbone = DINOv3ConvNeXtBackbone(
            checkpoint_source=args.dinov3_convnext_path,
            timm_model_name=timm_model_name,
            freeze_stages=args.freeze_convnext_stages,
        )
        args.features_dim = backbone.base_output_dim
        return backbone

    if args.backbone == "dinov3vits":
        backbone = DINOv3ViTBackbone(args.dinov3_vit_path)
        args.features_dim = backbone.base_output_dim
        return backbone

    if args.backbone.startswith("resnet"):#判断backbone的前面部分是不是resnet
        if args.pretrain in ["places", "gldv2"]:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            logging.info("Loading torchvision ImageNet pretrained resnet18")
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            logging.info("Loading torchvision ImageNet pretrained resnet50")
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            logging.info("Loading torchvision ImageNet pretrained resnet101")
            backbone = torchvision.models.resnet101(pretrained=True)
        if not args.unfreeze:
            for name, child in backbone.named_children():#遍历 ResNet 的直接子模块（如 conv1, bn1, layer1, layer2, layer3, layer4, avgpool, fc）
                # Freeze layers before conv_3
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False#将前面所有层的参数设为不可训练（冻结），节省显存、加快训练、防止过拟合
        if args.backbone.endswith("conv4"):#判断backbone的后面部分是不是conv4，如果配置使用到 conv4 为止（即去掉最后的 conv5 和全连接层）
            if not args.unfreeze:#记录当前配置（仅训练 conv4，移除 conv5）
                logging.debug(
                    f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x)")
            layers = list(backbone.children())[:-3]#ResNet 的最后 3 个子模块通常是 layer4 (conv5)、avgpool、fc，[:-3] 表示保留除了这 3 个之外的所有层，即网络输出到 layer3 (conv4)
        elif args.backbone.endswith("conv5"):#如果配置使用到 conv5 为止（去掉最后的 avgpool 和 fc
            if not args.unfreeze:
                logging.debug(
                    f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}, freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}")
            layers = list(backbone.children())[:-2]

        if args.remove_relu is True and (args.backbone.startswith("resnet50") or args.backbone.startswith("resnet101")):
            last_layer = layers[-1][-1]
            last_layer = nn.Sequential(*list(last_layer.modules())[1:-1])
            layers[-1][-1] = last_layer

    elif args.backbone == "swin_t":
        logging.warning("[DEBUG] Using Swin-T backbone branch WITH strict_img_size=False")
        use_pretrained = str(getattr(args, "pretrain", "imagenet")).lower() not in {
            "none",
            "null",
            "false",
            "random",
        }
        backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=use_pretrained,
            features_only=True,
            out_indices=(3,),
            img_size=None,
            dynamic_img_size=True,
            strict_img_size=False,
        )
        if hasattr(backbone, "layers"):
            for i, layer in enumerate(backbone.layers):
                if i < 3:
                    for p in layer.parameters():
                        p.requires_grad = False
        args.features_dim = 768
        return backbone
    #
    elif args.backbone.startswith("vit"):
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        else:
            raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te + 1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone


def get_output_channels_dim(model):
    """返回模型输出中的通道数。"""
    return model(torch.ones([1, 3, 224, 224])).shape[1]
