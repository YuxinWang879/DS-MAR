#把 backbone 输出的“空间特征图/Token 序列”聚合成一个“全局描述子 embedding
#network.py 里 GeoLocalizationNet 的 self.aggregation = get_aggregation(args) 最终拿到的具体实现，就在这个文件里定义
import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler

import model.functional as LF
import model.normalization as normalization

#Max pooling 聚合（全局最大池化）
class MAC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return LF.mac(x)
    def __repr__(self):
        return self.__class__.__name__ + '()'

#Sum pooling / average pooling 聚合（通常是全局平均池化的变体）。在检索里叫 SPoC
class SPoC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return LF.spoc(x)
    def __repr__(self):
        return self.__class__.__name__ + '()'

#Generalized Mean Pooling（可学习的 p），介于 avg 和 max 之间
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

#Regional MAC，多尺度多区域的 MAC 聚合（更强调区域信息）
class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super().__init__()
        self.L = L
        self.eps = eps
    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'

#把 (B,C,1,1) 压成 (B,C)
class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                聚类中心的数量
            dim : int
                描述符的维数
            alpha : float
                初始化参数。更大的值更难赋值
            normalize_input : bool
                是否对输入特征做 L2 归一化（默认 True）
            work_with_tokens：是否处理序列格式（Token）特征（如 Transformer 输出）
        """
        super().__init__()
        # 将参数保存为实例属性，方便后续方法使用
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0  # 初始化软分配参数，后续会通过 init_params 重新赋值
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        # 创建一个卷积层，用于计算每个特征点到聚类中心的软分配（soft assignment）
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))  # 定义聚类中心为可学习参数

    # 用 K-means 得到的聚类中心初始化 NetVLAD 参数
    def init_params(self, centroids, descriptors):#centroids：K-means 得到的聚类中心，descriptors：用于训练 K-means 的特征描述符
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)#对聚类中心做 L2 归一化，使其长度为 1
        #计算每个描述符与所有聚类中心的内积，按降序排序
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()#计算软分配参数 alpha
        self.centroids = nn.Parameter(torch.from_numpy(centroids))#将 K-means 得到的聚类中心转为 PyTorch 参数
        #用 alpha * 归一化聚类中心 初始化卷积层权重
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None#禁用偏置，保持与初始化逻辑一致

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)

        sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        loader = DataLoader(
            cluster_ds,
            batch_size=args.infer_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        # 允许 backbone 其实是整个 model（里面有 .backbone 和 .conv_layer）
        conv_layer = None
        real_backbone = backbone
        if hasattr(backbone, "backbone") and hasattr(backbone, "conv_layer"):
            conv_layer = backbone.conv_layer
            real_backbone = backbone.backbone
        all_desc = []
        with torch.no_grad():
            real_backbone.eval()
            if conv_layer is not None:
                conv_layer.eval()
            for inputs, _ in tqdm(loader, ncols=100):
                inputs = inputs.to(args.device)
                # feat = real_backbone(inputs)
                # if isinstance(feat, (list, tuple)):
                #     feat = feat[0]  # Swin stage4
                # elif hasattr(feat, "last_hidden_state"):
                #     feat = feat.last_hidden_state
                # if conv_layer is not None:
                #     feat = conv_layer(feat)
                feat = real_backbone(inputs)
                if isinstance(feat, (list, tuple)):
                    feat = feat[0]  # Swin stage4
                elif hasattr(feat, "last_hidden_state"):
                    feat = feat.last_hidden_state

                # Swin/timm features_only 可能输出 NHWC: [B, H, W, C]
                # Conv2d 需要 NCHW: [B, C, H, W]，所以必须在 conv_layer 之前转换
                if (not self.work_with_tokens) and feat.ndim == 4:
                    if feat.shape[1] < 32 and feat.shape[-1] >= 32:
                        feat = feat.permute(0, 3, 1, 2).contiguous()

                if conv_layer is not None:
                    feat = conv_layer(feat)


                if self.work_with_tokens:
                    if feat.ndim != 3:
                        raise RuntimeError(f"[NetVLAD init] Expected 3D token feat, got {feat.shape}")
                    B, HW, C = feat.shape
                    if C != self.dim:
                        raise RuntimeError(
                            f"[NetVLAD init] Token channel mismatch AFTER conv_layer: feat C={C}, NetVLAD dim(self.dim)={self.dim}."
                        )
                    feat = F.normalize(feat, p=2, dim=2)
                    desc = feat.cpu().numpy()
                    HW = desc.shape[1]
                    k = min(descs_num_per_image, HW)
                    for i in range(B):
                        idx = np.random.choice(HW, k, replace=False)
                        all_desc.append(desc[i, idx])
                    continue
                if feat.ndim != 4:
                    raise RuntimeError(f"[NetVLAD init] Expected 4D feat, got {feat.shape}")
                # ---- NHWC -> NCHW（非常关键，避免 C=8 这种把 H 当 C 的错误）----
                # 如果第二维不是通道，但最后一维看起来像通道（>=32 更像通道），就转
                if feat.shape[1] < 32 and feat.shape[-1] >= 32:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                # ---- 过 conv_layer：让通道数对齐 NetVLAD 的 self.dim（例如 64）----
                B, C, H, W = feat.shape
                if C != self.dim:
                    raise RuntimeError(
                        f"[NetVLAD init] Channel mismatch AFTER conv_layer: feat C={C}, NetVLAD dim(self.dim)={self.dim}. "
                        f"说明你的 conv_layer 输出通道数和 NetVLAD dim 没配一致。"
                    )
                feat = F.normalize(feat, p=2, dim=1)
                desc = feat.view(B, C, -1).permute(0, 2, 1).cpu().numpy()  # [B, HW, C]
                HW = desc.shape[1]
                k = min(descs_num_per_image, HW)
                for i in range(B):
                    idx = np.random.choice(HW, k, replace=False)
                    all_desc.append(desc[i, idx])
        descriptors = np.concatenate(all_desc, axis=0).astype(np.float32)
        if descriptors.shape[0] > descriptors_num:
            sel = np.random.choice(descriptors.shape[0], descriptors_num, replace=False)
            descriptors = descriptors[sel]
        # kmeans 维度必须等于 self.dim（不要再用 args.features_dim 兜底乱变）
        kmeans = faiss.Kmeans(self.dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
