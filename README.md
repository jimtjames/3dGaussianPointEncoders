# 3D Gaussian Point Encoders (WACV 2026)
Jim James, Benjamin Wilson, Simon Lucey, James Hays

This repository contains the official implementation corresponding to the paper "3D Gaussian Point Encoders," published at WACV 2026 (see [here](https://openaccess.thecvf.com/content/WACV2026/papers/James_3D_Gaussian_Point_Encoders_WACV_2026_paper.pdf)).

## Usage

### Clone

```
git clone jimtjames/3dGaussianPointEncoders
```

Experiments are split between the `PointNet` directory (for PointNet-style models with MLP classifiers) and the `Mamba3D` (for Mamba3D with a 3DGPE patch encoder).

### Dependencies and Environment

The dependencies for the PointNet and Mamba3D experiments slightly differ, due to specifics of Mamba3D's mamba-ssm compatibility. Accordingly, they have distinct python environments. Broadly, we use Python 3.12 with CUDA 13. We manage the dependencies of our experiments with [Nix](https://nixos.org). On non-NixOS machines, you may need to prepend `nixglhost` before running python commands to ensure interoperability with your system's CUDA drivers.

To build the dependencies, go to the `nix` subdirectory and run `nix develop`. After building, this will drop you into a dev shell (similar to a venv, but with some dependencies built from nix packages instead). Be aware that this can take quite a while and use a lot of RAM if building from scratch.

As an alternative, we also include a `uv.lock` file. However, this is less tested as we didn't run experiments with it.

### Data Preparation

Place copies of both ModelNet40 and ScanObjectNN data in PointNet/data and Mamba3D/data. See Mamba3D/DATASET.md for details.

###  Running Experiments

To run PointNet experiments, use the following scripts for examples with different natural gradients, e.g..

```
# Mahalanobis natural gradient
python3 train.py --n 32 --batch_size 32 --dataset scanobjnn_hard --e2e --dual_optim --hook mahalanobis 

# Fischer natural gradient
python3 train.py --n 32 --batch_size 32 --dataset scanobjnn_hard --e2e --dual_optim --hook mahalanobis 
```

For Mamba3D Experiments, see the configs under Mamba3D/cfgs for examples with different natural gradient variants, e.g.:

```
# Mahalanobis natural gradient

python main.py --config cfgs/finetune_scan_hardest_mahalanobis_32_dual.yaml --scratch_model --exp_name GaussianMamba3D_hardest_mahalanobis_32_dual_scratch --resume

# Fischer natural gradient
python main.py --config cfgs/finetune_scan_hardest_fim_32_dual.yaml --scratch_model --exp_name GaussianMamba3D_hardest_mahalanobis_32_dual_scratch --resume
```

Our training experiments were predominantly run on a NixOS 25.05 machine with NVIDIA GeForce RTX 5090 GPU and CUDA 13.

## Citation

If you found this repository useful, please cite our paper with the following:

```
@InProceedings{James_2026_WACV,
    author    = {James, Jim and Wilson, Benjamin and Lucey, Simon and Hays, James},
    title     = {3D Gaussian Point Encoders},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {1788-1797}
}
```
