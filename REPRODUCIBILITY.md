# Reproducibility Guide

This document describes exactly how to reproduce the main results of this
project, and documents the environment, versions, and random seeds used.

## Environment

| Component | Version / Spec |
|---|---|
| Python | 3.13.15 |
| PyTorch (local) | 2.14.0+cpu |
| Local hardware | AMD Ryzen 5 7430U (6 cores/12 threads), 8 GB RAM, integrated graphics (no CUDA) |
| Training hardware (fine-tuning) | Google Colab, Tesla T4 GPU (free tier) |
| OS | Windows 11 |

See `requirements.txt` for the full pinned dependency list.

## Dataset

- **Source:** EuroSAT RGB, official Zenodo release
- **DOI:** 10.5281/zenodo.7711810
- **License:** MIT
- **Download:** `python src/data/download_dataset.py` (verifies MD5 checksum
  `f46e308c4d50d4bf32fedad2d3d62f3b` automatically)

## Random Seed

All stochastic steps use a fixed seed of **42**, set explicitly in:
- `src/data/create_split.py` (train/val/test split)
- `src/training/train_baseline.py`, `train_transfer.py`, `train_finetune.py`
  (`torch.manual_seed(42)`)

Re-running the full pipeline with this repository's code should reproduce
an equivalent train/val/test split and directly comparable training
dynamics, though exact bit-for-bit reproducibility of GPU-trained results
is not guaranteed (CUDA operations are not fully deterministic by default).

## Reproducing the Full Pipeline

Run in order from the project root, with the virtual environment activated:

```bash
# 1. Environment setup (see README for full details)
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2. Dataset acquisition and verification
python src/data/download_dataset.py
python src/data/verify_dataset.py

# 3. Train/val/test split
python src/data/create_split.py

# 4. Normalization statistics (train split only)
python src/data/compute_normalization.py

# 5. Baseline CNN
python src/training/train_baseline.py
python src/evaluation/evaluate_baseline.py

# 6. Transfer learning — frozen backbone
python src/training/train_transfer.py
python src/evaluation/evaluate_transfer.py

# 7. Transfer learning — fine-tuning (GPU strongly recommended; see note below)
python src/training/train_finetune.py
python src/evaluation/evaluate_finetune.py

# 8. Package final model
python src/models/save_final_model.py

# 9. Error analysis and explainability
python src/evaluation/error_analysis.py
python src/explainability/generate_gradcam_examples.py

# 10. Run tests
pytest tests/ -v

# 11. Launch the app
streamlit run app/app.py
```

### Note on hardware and step 7

Fine-tuning (`train_finetune.py`) unfreezes the full ResNet18 backbone
(~11.2M trainable parameters) and is computationally expensive. In this
project:
- Local CPU (AMD Ryzen 5 7430U): not attempted for the full run after
  initial per-epoch timing (~18+ min/epoch) made 10 epochs impractical.
- Colab T4 GPU: ~85 seconds/epoch, ~14 minutes total for 10 epochs.

**Recommendation:** use a CUDA-capable GPU (local or via Colab) for this
step. The baseline (Phase 7) and frozen-backbone (Phase 8) training steps
are tractable on CPU (~25 min and ~2.5 hr respectively, on the hardware
above) if GPU access is unavailable.

## Training Times Summary

| Model | Epochs | Hardware | Time |
|---|---|---|---|
| Baseline CNN | 15 | Local CPU | ~25 min |
| ResNet18 (frozen) | 15 | Local CPU | ~2.5 hr |
| ResNet18 (fine-tuned) | 10 | Colab T4 GPU | ~14 min |

## Results Summary

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Baseline CNN | 92.15% | 0.9192 |
| ResNet18 (frozen) | 93.21% | 0.9295 |
| **ResNet18 (fine-tuned) — final model** | **97.93%** | **0.9787** |

## Final Model Artifacts

- `models/final_model.pth` — trained weights
- `models/model_config.json` — class mapping, input size, normalization
  constants, and training summary metadata

These two files together are sufficient to run inference without access
to the training pipeline or dataset — see `src/inference/predictor.py`.