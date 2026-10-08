"""Minimal training entry point for the released SAMF model."""

import logging
import math
import multiprocessing
import time
from datetime import datetime
from os.path import join
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import commons
import datasets_ws
import parser
import test
import util
from model import network


logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.benchmark = True


def _modalities(batch_size, negatives_per_query, device):
    """Return 1 for thermal queries and 0 for satellite images."""
    group_size = 2 + negatives_per_query
    modality = torch.zeros(batch_size * group_size, dtype=torch.long, device=device)
    modality[::group_size] = 1
    return modality


def _triplet_loss(features, local_indexes, batch_size, negatives_per_query, criterion):
    triplets = local_indexes.view(batch_size, negatives_per_query, 3).transpose(0, 1)
    loss = 0.0
    for batch_triplets in triplets:
        query_idx, positive_idx, negative_idx = batch_triplets.T
        loss = loss + criterion(
            features[query_idx], features[positive_idx], features[negative_idx]
        )
    return loss / (batch_size * negatives_per_query)


def main():
    args = parser.parse_arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable; falling back to CPU.")
        args.device = "cpu"
    start_time = datetime.now()
    args.save_dir = join(
        "logs",
        args.save_dir,
        f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
    )
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info("Arguments: %s", args)
    logging.info(
        "Using %d GPU(s) and %d CPU(s)",
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        multiprocessing.cpu_count(),
    )

    train_ds = datasets_ws.TripletsDataset(
        args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query
    )
    val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.info("Train set: %s", train_ds)
    logging.info("Validation set: %s", val_ds)
    logging.info("Test set: %s", test_ds)

    model = network.GeoLocalizationNet(args).to(args.device)

    # NetVLAD needs a one-time initialization from training images.
    train_ds.is_inference = True
    model.aggregation.initialize_netvlad_layer(args, train_ds, model)
    args.features_dim *= args.netvlad_clusters
    train_ds.is_inference = False
    # NetVLAD initialization recreates its centroids and assignment layer on
    # the default device. Move the complete model back to the selected device.
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
    best_r5 = 0.0
    not_improved = 0

    for epoch in range(args.epochs_num):
        logging.info("Start training epoch %02d", epoch)
        epoch_start = time.perf_counter()
        epoch_losses = []
        loops = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)

        for loop in range(loops):
            train_ds.is_inference = True
            train_ds.compute_triplets(args, model, epoch_num=epoch)
            train_ds.is_inference = False
            loader = DataLoader(
                train_ds,
                batch_size=args.train_batch_size,
                num_workers=args.num_workers,
                collate_fn=datasets_ws.collate_fn,
                drop_last=True,
            )

            model.train()
            for images, local_indexes, _ in tqdm(loader, ncols=100):
                modalities = _modalities(
                    args.train_batch_size,
                    args.negs_num_per_query,
                    args.device,
                )
                features = model(
                    images.to(args.device),
                    is_train=True,
                    modality=modalities,
                )
                loss = _triplet_loss(
                    features,
                    local_indexes,
                    args.train_batch_size,
                    args.negs_num_per_query,
                    criterion,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            del loader

        logging.info(
            "Finished epoch %02d | loss=%.6f | time=%.1fs",
            epoch,
            float(np.mean(epoch_losses)),
            time.perf_counter() - epoch_start,
        )

        model.eval()
        recalls, recalls_str = test.test(args, val_ds, model)
        logging.info("Validation recalls: %s", recalls_str)
        is_best = recalls[1] > best_r5
        util.save_checkpoint(
            args,
            {
                "epoch_num": epoch,
                "model_state_dict": model.state_dict(),
                "model_db_state_dict": None,
                "DA_state_dict": None,
                "optimizer_state_dict": optimizer.state_dict(),
                "recalls": recalls,
                "best_r5": best_r5,
                "not_improved_num": not_improved,
            },
            is_best,
            filename="last_model.pth",
        )
        if is_best:
            best_r5 = recalls[1]
            not_improved = 0
            logging.info("New best model: epoch=%02d, R@5=%.1f", epoch, best_r5)
        else:
            not_improved += 1
            if not_improved >= args.patience:
                logging.info("Early stopping after %d unimproved epochs", not_improved)
                break

    checkpoint = torch.load(join(args.save_dir, "best_model.pth"), map_location=args.device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    recalls, recalls_str = test.test(
        args, test_ds, model, test_method=args.test_method
    )
    logging.info("Final test recalls: %s", recalls_str)


if __name__ == "__main__":
    main()
