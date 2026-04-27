# DeblurDiff: Real-World Image Deblurring with Generative Diffusion Models
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.03810)

## Dependencies

The checked-in environment targets Ubuntu 20.04 / A100 with Python 3.10,
PyTorch 2.1.2, and CUDA 12.1:

```bash
mamba env create -f environment.yml
conda activate deblurdiff-cu121
```

`conda` can be used instead of `mamba`, but dependency solving is usually
slower:

```bash
conda env create -f environment.yml
conda activate deblurdiff-cu121
```

Verify the CUDA stack before running inference or training:

```bash
python - <<'PY'
import torch, cupy
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
cupy.show_config()
PY
```

## checkpoint 

[download](https://drive.google.com/drive/folders/1CUtnUKbu_zTyjJ17F95UYyh2SDzCOHeW?usp=drive_link)

## Data
run bash file_img.sh to generate the training data list.

## Train

bash train.sh

## Test

bash test.sh

## Acknowledgment: 

This code is based on the [DiffBIR](https://github.com/XPixelGroup/DiffBIR) and [DemystifyLocalViT](https://github.com/Atten4Vis/DemystifyLocalViT). Thanks for their awesome work.
