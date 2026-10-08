"""Command-line options for the public SAMF training recipe.

This repository exposes one reproducible training path: Swin-T + NetVLAD +
triplet loss with DSEM and MARM. The defaults below are the settings used by
the released experiment; only dataset, runtime, and training-length options
are configurable from the command line.
"""

import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train SAMF on the thermal-to-satellite dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets_folder", default=None,
                        help="Directory containing the dataset folder.")
    parser.add_argument("--dataset_name", default="satellite-thermal-dataset-v1",
                        help="Dataset folder name.")
    parser.add_argument("--save_dir", default="samf_swin_dsm_marm",
                        help="Subdirectory created under ./logs.")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256],
                        metavar=("HEIGHT", "WIDTH"), help="Input image size.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs_num", type=int, default=30)
    parser.add_argument("--queries_per_epoch", type=int, default=10000)
    parser.add_argument("--cache_refresh_rate", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--prior_location_threshold", type=int, default=-1)
    parser.add_argument("--val_positive_dist_threshold", type=int, default=50)
    parser.add_argument("--train_positives_dist_threshold", type=int, default=35)
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20])
    args = parser.parse_args()

    if args.datasets_folder is None:
        args.datasets_folder = os.environ.get("DATASETS_FOLDER")
    if not args.datasets_folder:
        parser.error("--datasets_folder is required (or set DATASETS_FOLDER).")
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        parser.error("queries_per_epoch must be divisible by cache_refresh_rate.")

    # Fixed SAMF architecture options.
    args.backbone = "swin_t"
    args.aggregation = "netvlad"
    args.netvlad_clusters = 64
    args.conv_output_dim = 8192
    args.add_bn = True
    args.l2 = "before_pool"
    args.pretrain = "imagenet"
    args.off_the_shelf = "imagenet"
    args.criterion = "triplet"
    args.optim = "adam"
    args.weight_decay = 0.0
    args.backbone_lr_mult = 1.0
    args.negs_num_per_query = 10
    args.neg_samples_num = 1000
    args.mining = "partial"
    args.use_faiss_gpu = False
    args.use_best_n = 1
    args.test_method = "hard_resize"
    args.majority_weight = 0.01
    args.efficient_ram_testing = False
    args.compute_map = False
    args.map_k = 1000
    args.resume = None
    args.freeze_backbone_bn = False
    args.unfreeze = False
    args.remove_relu = False
    args.non_local = False
    args.num_non_local = 1
    args.channel_bottleneck = 128
    args.fc_output_dim = None
    args.trunc_te = None
    args.freeze_te = None
    args.work_with_tokens = False
    args.DA = "none"

    # DSEM and MARM settings used in the paper.
    args.use_dsm = True
    args.dsm_num_dirs = 8
    args.dsm_low_kernel = 5
    args.dsm_use_semantic_branch = 1
    args.dsm_use_directional_branch = 1
    args.dsm_use_low_frequency_branch = 1
    args.lambda_dir = 0.0
    args.lambda_struct_triplet = 0.0
    args.struct_margin = 0.05
    args.use_modality_affine_remap = True
    args.modality_affine_sigma_gamma = 0.02
    args.modality_affine_sigma_beta = 0.01

    # Dataset preprocessing used by this recipe.
    args.G_contrast = True
    args.G_gray = False
    args.brightness = None
    args.contrast = None
    args.saturation = None
    args.hue = None
    args.rand_perspective = None
    args.horizontal_flip = False
    args.random_resized_crop = None
    args.random_rotation = None

    # Evaluation switches retained as fixed values for train.py.
    args.G_test_norm = "batch"
    args.features_dim = None
    return args
