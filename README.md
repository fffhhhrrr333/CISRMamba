# 🌊 CISRMamba

> **CISRMamba: Cross-modal Interaction and Scan Routing Mamba for Optical-SAR Flood Detection**

✨ A dual-stream Mamba network for **flood / water-body segmentation** from
co-registered **optical (4-channel)** and **SAR (1-channel, VV)** Sentinel
imagery.

The four letters of **CISR** map one-to-one to the core contributions 👇

| Letter | Expansion | Module |
|--------|-----------|--------|
| 🔀 **C** | **C**ross-modal | FCM — three-way (enhance / complement / discard) gate over optical ↔ SAR |
| 🔗 **I** | **I**nteraction | RFI — spatial gate + SE channel recalibration + cross-modal diff residual |
| 🌀 **S** | **S**can | SS4D — 4-directional state-space scanning |
| 🧭 **R** | **R**outing | DeformableScanRouter — learns per-scale offsets that steer SS4D scans |

---

## 📁 1. Directory layout

```
code/
├── 🧠 CISRMamba.py        # Main model: dual-stream backbone + FCM + RFI + decoder
├── 🦎 backbone.py         # MobileMamba / VSS encoder with deformable scan routing
├── 🔌 encoder.py          # Thin wrapper that exposes `VSSEncoder`
├── 📦 lib_mamba/          # Mamba SSM primitives (SS2D/SS4D, selective scan, CSM kernels)
├── 🗂️  dataloaded.py       # CustomDataset (4-ch opt + 1-ch SAR + binary mask)
├── ⚙️  config.py           # Hyper-parameters and dataset / checkpoint paths
├── 🧰 util.py             # train_fn / eval_fn, metrics, visualization helpers
├── 📉 loss.py             # Dice + BCE loss
├── 📏 evaluate.py         # Stand-alone metric functions (Acc / Sens / Spec / ...)
├── 🏋️  train.py            # Training entry point
├── 🧪 test.py             # Evaluation entry point (metrics + 1 sample vis)
├── 🖼️  picture.py          # Batch inference: save a prediction mask per test image
├── 📜 requirements.txt
└── 📖 README.md
```

---

## 🛠️ 2. Environment

- 🐍 Python 3.8
- 🟢 CUDA 11.8, NVIDIA GPU with ≥ 8 GB memory (tested on a single RTX 3090)
- 🐧 Linux (Ubuntu 20.04) — Mamba kernels require Linux + CUDA

Install with 👇

```bash
conda create -n cisrmamba python=3.8 -y
conda activate cisrmamba
pip install -r requirements.txt
```

> ⚠️ `mamba-ssm==1.1.1` and `causal-conv1d==1.1.1` contain pre-built CUDA
> extensions. If pip cannot find a matching wheel for your platform you may
> need to `pip install --no-build-isolation` and ensure `nvcc` / `CUDA_HOME`
> are set.

---

## 🗃️ 3. Dataset layout

The provided `CustomDataset` expects the three modalities to share the **same
filename** across three folders:

```
<root>/
├── 🌈 opt/        # 4-channel optical (e.g. B-G-R-NIR), 8-bit or 16-bit
├── 📡 vv/         # 1-channel SAR VV
└── 💧 flood_vv/   # binary flood mask (0 / 255)
```

🧭 Configure training / test roots in `config.py` (`OPTICAL_DIR`, `RADAR_DIR`,
`LABEL_DIR`) and in `test.py` / `picture.py` (`TEST_OPT_DIR`, `TEST_SAR_DIR`,
`TEST_LBL_DIR`).

📐 All images are resized to `height × width = 256 × 256` by the Albumentations
pipeline before being concatenated into a 5-channel tensor.

---

## 🏋️ 4. Training

Edit `config.py` as needed (`EPOCHS`, `BATCH_SIZE`, `LR`, paths), then 🚀

```bash
python train.py
```

The script:

- ✂️ Splits the dataset 80 / 20 with a fixed seed (42) for train / val
- ⚡ Optimizer: **AdamW** (`lr=1e-4`, `weight_decay=0.01`)
- 🌀 Scheduler: **CosineAnnealingLR** (`T_max=EPOCHS`, `eta_min=1e-6`)
- 🎯 Loss: Dice + BCE-with-logits (computed inside `CISRMamba.forward`)
- 💾 Saves `last_model.pt` every epoch and `CISRMamba_scratch_CISRMamba_best.pt`
  whenever mIoU improves, to `CHECKPOINT_DIR`
- 📊 Logs per-epoch metrics (mIoU / F1 / Acc / Prec / Recall / Kappa) to
  `METRICS_CSV`

---

## 🧪 5. Evaluation

`test.py` computes the full metric set on a separate test directory and also
reports **params / GFLOPs** (requires `thop`). 📏

```bash
python test.py
```

🔧 Adjust the paths at the top of `test.py`:

```python
CHECKPOINT_PATH = "/CISRMamba/final/CISRMamba_scratch_CISRMamba_best.pt"
TEST_OPT_DIR    = "/test/opt"
TEST_SAR_DIR    = "/test/vv"
TEST_LBL_DIR    = "/test/flood_vv"
```

---

## 🖼️ 6. Exporting prediction masks

`picture.py` runs inference over **every** test sample and saves a binary PNG
(0 / 255) per image, reusing the original filename:

```bash
python picture.py
```

📤 Output directory: `RESULT_IMG_DIR = "/root/autodl-tmp/test_result"` (edit at
the top of the file).

---

## 🧩 7. Key modules

| 🧱 Module | 📄 File | 🎯 Role |
|--------|------|------|
| 🦎 `VSSEncoder` / `MobileMambaEncoder` | `backbone.py`, `encoder.py` | Hierarchical Mamba encoder, 4 scales, dims `[96, 192, 384, 768]`, with **DeformableScanRouter** controlling **SS4D** |
| 🔀 `FCM` | `CISRMamba.py` | Three-way gate (enhance / complement / discard) over optical ↔ SAR, with bidirectional cross-attention on spatially reduced keys/values |
| 🔗 `RFI` | `CISRMamba.py` | Spatial gate + SE channel recalibration + cross-modal diff residual |
| 🎛️ `CEB` | `CISRMamba.py` | Channel Enhanced Block used inside `ResBlock_CBAM` |
| 🌊 `SRB` / `SFE` | `CISRMamba.py` | Haar DWT + FreMLP (magnitude / phase) refinement at the final decoder output |
| 🧭 `AlignDecoderBlock` + `CAB` | `CISRMamba.py` | Deformation-aware decoder that aligns low- and high-level features via learned offsets |

---

## 📚 8. Citation / acknowledgement

🙏 This code builds on the **MobileMamba / SegMAN** backbone and on
`segmentation-models-pytorch` for the Dice loss. Please cite the respective
works if you use this repository in your research.

---

<p align="center">⭐ If this repo helps your work, please give it a star! ⭐</p>
