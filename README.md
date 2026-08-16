<div align="center">

# CISRMamba

### Cross-Modal Interaction and Scan-Routing Mamba for Multi-Sensor Flood Inundation Mapping

[![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Weights](https://img.shields.io/badge/Model-Weights-4C8BF5)](https://github.com/fffhhhrrr333/CISRMamba/releases/tag/1)

Official PyTorch implementation of **CISRMamba**, a dual-stream state-space model for
flood-inundation mapping from co-registered optical and SAR observations.

[Overview](#-overview) &nbsp;·&nbsp; [Architecture](#-architecture) &nbsp;·&nbsp;
[Results](#-results) &nbsp;·&nbsp; [Installation](#installation) &nbsp;·&nbsp;
[Weights](#-pretrained-weights) &nbsp;·&nbsp; [Evaluation](#evaluation)

</div>

## 🔎 Overview

CISRMamba addresses two recurring challenges in optical--SAR fusion: spatially varying
cross-modal reliability and the difficulty of tracing irregular flood boundaries with fixed
scan patterns. The network combines four components:

- **Deformable Scan Router (DSR):** adapts state-space scan trajectories to irregular flood contours.
- **Feature Complementary Module (FCM):** routes local responses through enhancement, complementation, or discrepancy suppression.
- **Refined Feature Integration (RFI):** combines spatial gating, channel recalibration, and a cross-modal difference residual.
- **Spectral Refinement Block (SRB):** restores high-frequency boundary details with wavelet-domain refinement.

The name **CISR** summarizes the central design:

| Initial | Principle | Implementation |
|:--:|:--|:--|
| 🔀 **C** | Cross-modal | FCM performs region-dependent routing between optical and SAR features. |
| 🔗 **I** | Interaction | RFI preserves complementary information while limiting modality dominance. |
| 🌀 **S** | Scan | SS4D provides four-directional state-space context modeling. |
| 🧭 **R** | Routing | DSR learns spatial offsets that steer the scan toward irregular structures. |

## 🧩 Architecture

<p align="center">
  <img src="https://github.com/fffhhhrrr333/CISRMamba/releases/download/1/cisrmamba_architecture.png" width="100%" alt="Overall architecture of CISRMamba">
</p>

The optical and SAR streams use structurally identical but independently parameterized
four-stage encoders. DSR precedes the state-space scan at each scale; FCM and RFI then
perform region-dependent cross-modal fusion. A progressive decoder and the final SRB
produce the binary flood map.

### Key components

| Component | Location | Role |
|:--|:--|:--|
| `VSSEncoder` | `backbone.py`, `encoder.py` | Four-scale optical and SAR feature extraction. |
| `DeformableScanRouter` | `backbone.py` | Content-adaptive feature warping before state-space scanning. |
| `FCM` | `CISRMamba.py` | Enhancement, complementation, and discrepancy-suppression routing. |
| `RFI` | `CISRMamba.py` | Spatial and channel recalibration with a difference-guided residual. |
| `SRB` / `SFE` | `CISRMamba.py` | Wavelet- and frequency-domain boundary refinement. |
| `AlignDecoderBlock` | `CISRMamba.py` | Progressive alignment and decoding of multi-scale features. |

## 📊 Results

### Quantitative comparison

The following results are reported in the paper. All values are percentages.

| Dataset | mIoU | Accuracy | F1 | Precision | Recall |
|:--|--:|--:|--:|--:|--:|
| CAU-Flood | **91.53** | **97.81** | **95.47** | **95.23** | **95.73** |
| Wuhan | **58.67** | **87.67** | **70.65** | **72.89** | **68.56** |

On CAU-Flood, CISRMamba also obtains **86.03% Water IoU**. The complete model contains
**53.51 M parameters** and requires **13.95 GMACs** for a 256 x 256 input pair.

### Qualitative comparison on CAU-Flood

<p align="center">
  <img src="https://github.com/fffhhhrrr333/CISRMamba/releases/download/1/cau_flood_qualitative_results.png" width="82%" alt="Qualitative comparison on the CAU-Flood dataset">
</p>

The red boxes highlight narrow channels, isolated water bodies, and regions with strong
background interference. White and black denote true positives and true negatives;
blue and magenta denote false positives and false negatives, respectively.

### Component ablation on CAU-Flood

<p align="center">
  <img src="https://github.com/fffhhhrrr333/CISRMamba/releases/download/1/cau_flood_ablation.png" width="100%" alt="Qualitative component ablation on the CAU-Flood dataset">
</p>

Each row shows the optical image, SAR image, ground truth, four single-component-removal
variants, and the complete model. Differences are concentrated around narrow channels,
curved water--land boundaries, and isolated flood patches.

## Installation

The reference environment uses Python 3.8, PyTorch 2.0.0, and CUDA 11.8 on Linux.
A CUDA-capable GPU is required by the selective-scan kernels.

```bash
conda create -n cisrmamba python=3.8 -y
conda activate cisrmamba
pip install -r requirements.txt
```

If a compatible wheel is unavailable for `mamba-ssm` or `causal-conv1d`, install the
packages against the local CUDA toolkit with build isolation disabled.

## Data preparation

The data loader expects co-registered optical images, SAR images, and binary masks to
share the same filename:

```text
<dataset-root>/
|-- opt/       # four-channel optical images
|-- vv/        # single-channel SAR images
`-- flood_vv/  # binary masks
```

Set `OPTICAL_DIR`, `RADAR_DIR`, and `LABEL_DIR` in `config.py`. Inputs are converted to
floating point, scaled from 0--255 to 0--1, and resized to 256 x 256. The training pipeline
applies independent horizontal and vertical flips with probability 0.5; validation does
not use augmentation.

> [!IMPORTANT]
> CISRMamba assumes that each optical--SAR pair already covers the same geographic extent
> and has been registered to a common grid. The model does not perform image registration.

## 📦 Pretrained weights

Download `CISRMamba_scratch_CISRMamba_best.pt` from the
[GitHub release](https://github.com/fffhhhrrr333/CISRMamba/releases/tag/1):

```bash
wget https://github.com/fffhhhrrr333/CISRMamba/releases/download/1/CISRMamba_scratch_CISRMamba_best.pt
```

Alternatively, use `curl`:

```bash
curl -L -o CISRMamba_scratch_CISRMamba_best.pt \
  https://github.com/fffhhhrrr333/CISRMamba/releases/download/1/CISRMamba_scratch_CISRMamba_best.pt
```

The released checkpoint was trained from scratch; no external pretrained weights were used.

## Training

Configure the dataset and output paths in `config.py`, then run:

```bash
python train.py
```

The reference configuration uses 60 epochs, batch size 16, AdamW with an initial learning
rate of `1e-4` and weight decay `0.01`, and cosine annealing to `1e-6`. The objective gives
equal weight to binary cross-entropy and Dice losses:

```text
L_total = 0.5 * L_BCE + 0.5 * L_Dice
```

During training, the code:

- saves `last_model.pt` after every epoch;
- saves `CISRMamba_scratch_CISRMamba_best.pt` whenever mIoU improves; and
- records losses and segmentation metrics in the CSV path specified by `METRICS_CSV`.

## Evaluation

Set the checkpoint, test-data, visualization, and CSV paths at the top of `test.py`:

```python
CHECKPOINT_PATH = "/CISRMamba/final2/CISRMamba_scratch_CISRMamba_best.pt"
TEST_OPT_DIR = "/test/opt"
TEST_SAR_DIR = "/test/vv"
TEST_LBL_DIR = "/test/flood_vv"
RESULT_IMG_PATH = "/path/to/visualization.png"
CSV_SAVE_PATH = "test_results_standalone.csv"
```

Then run:

```bash
python test.py
```

The script reports mIoU, accuracy, F1, precision, recall, parameter count, and computational
cost, and saves the evaluation summary to CSV.

## Repository structure

```text
.
|-- CISRMamba.py      # network, cross-modal modules, and decoder
|-- backbone.py       # MobileMamba encoder and deformable scan routing
|-- encoder.py        # encoder wrapper
|-- lib_mamba/        # selective-scan and state-space building blocks
|-- dataloaded.py     # paired optical--SAR dataset pipeline
|-- loss.py           # equally weighted BCE--Dice objective
|-- train.py          # training entry point
|-- test.py           # evaluation entry point
|-- evaluate.py       # segmentation metrics
|-- util.py           # training and evaluation utilities
|-- config.py         # experiment configuration
`-- requirements.txt  # Python dependencies
```

## Acknowledgements

This implementation builds on MobileMamba and `segmentation-models-pytorch`. We thank
the authors of the associated open-source projects and datasets.
