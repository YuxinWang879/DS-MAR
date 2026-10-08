# Structure-Aware Matching for Robust UAV Thermal-to-Satellite Geo-Localization
SAMF addresses cross-modal matching between thermal infrared UAV images and RGB satellite images. It contains a Directional Structure Enhancement Module (DSEM) and a Modality-Aware Affine Remapping Module (MARM).

## Requirements

The code was tested in the following Linux environment:

```text
Python       3.10.19
PyTorch      2.9.1+cu128
torchvision  0.24.1+cu128
timm         1.0.22
CUDA         12.8
```

Create the environment with:

```bash
conda env create -f env.yml
conda activate lth
```

## Datasets

The code supports both datasets used in the paper. The loader automatically selects the data format from `--dataset_name`:

- `satellite-thermal-dataset-v1`: Boson-nighttime HDF5 dataset.
- `thermal-uav-paired`: Thermal-UAV-Paired image dataset.

### Boson-nighttime

Boson-nighttime contains 10,256 training pairs, 13,011 validation pairs, and 26,568 test pairs. Each pair contains one thermal UAV image and one RGB satellite image stored in HDF5 files:

```text
datasets/
└── satellite-thermal-dataset-v1/
    ├── train_database.h5
    ├── train_queries.h5
    ├── val_database.h5
    ├── val_queries.h5
    ├── test_database.h5
    └── test_queries.h5
```

### Thermal-UAV-Paired

Thermal-UAV-Paired is the paired localization benchmark constructed from the Thermal-UAV dataset. It contains 7,886 training pairs, 1,407 validation pairs, and 2,350 test pairs. Each pair contains one thermal UAV image and one RGB satellite image. The image files are referenced by `pairs.json`:

```text
datasets/
└── thermal-uav-paired/
    ├── train/
    │   ├── pairs.json
    │   └── image files
    ├── valid/
    │   ├── pairs.json
    │   └── image files
    └── test/
        ├── pairs.json
        └── image files
```

Each record in `pairs.json` must provide `thermal` and `satellite` image paths. For prior-location evaluation, records should also provide `center_map_x` and
`center_map_y`, or `center_lon` and `center_lat`.

## Training

The public entry point uses the paper configuration by default. The fixed model configuration is Swin-T, 256x256 inputs, NetVLAD with 64 clusters, triplet loss, DSEM with eight directions and a 5x5 low-frequency kernel, and MARM.

Validation is run after each epoch. The final test set is evaluated using the best validation checkpoint, which is saved as `best_model.pth` under
`logs/<save_dir>/<dataset-and-run-id>/`.

## Citation

If you use this code or the Thermal-UAV-Paired benchmark, please cite the corresponding SAMF paper.

