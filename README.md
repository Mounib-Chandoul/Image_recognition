# 🧠 Image Recognition

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-9ECE6A?style=for-the-badge)](LICENSE)

</div>

A clean, framework-free image-classification project built with PyTorch. It
trains a small convolutional neural network on MNIST or on your own
`ImageFolder` dataset, evaluates after every epoch, saves a rich checkpoint,
and predicts from a single image.

---

## What it does

- Downloads and loads MNIST, or trains on a custom folder-based dataset.
- Normalizes images into tensors with a reusable transform pipeline.
- Defines a 2-convolution CNN with max pooling, dropout and adaptive pooling.
- Trains and evaluates end-to-end, with CPU/CUDA selection and seeded runs.
- Saves checkpoints containing weights, optimizer state, metrics and metadata.
- Runs single-image inference with restored class labels and preprocessing.

---

## Layout

```text
Image_recognition/
├── data_load.py      # transforms, MNIST / ImageFolder datasets, loaders
├── SimpleNetwork.py  # SimpleNet CNN definition
├── main.py           # training + evaluation CLI entry point
├── predict.py        # single-image inference from a checkpoint
├── PROJECT_REPORT.md # detailed project report
└── requirements.txt
```

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train on MNIST:

```bash
python main.py
```

Quick smoke run on a subset:

```bash
python main.py --epochs 2 --train-limit 5000 --test-limit 1000
```

Train on your own images:

```bash
python main.py --dataset imagefolder --data-dir path/to/dataset --epochs 10
```

Expected folder layout:

```text
path/to/dataset/
├── train/
│   ├── class_a/img1.jpg
│   └── class_b/img1.jpg
└── test/
    ├── class_a/img1.jpg
    └── class_b/img1.jpg
```

Predict one image:

```bash
python predict.py path/to/image.jpg --checkpoint checkpoints/mnist_cnn.pt
```

---

## CLI options (`main.py`)

| Flag | Purpose |
|---|---|
| `--dataset` | `mnist` or `imagefolder` |
| `--data-dir` | Dataset root (default `data`) |
| `--epochs` / `--batch-size` / `--learning-rate` | Training configuration |
| `--device` | Force `cpu` or `cuda` |
| `--seed` | Reproducible runs (default 42) |
| `--train-limit` / `--test-limit` | Small subsets for quick experiments |
| `--output` | Checkpoint path (default `checkpoints/mnist_cnn.pt`) |
| `--image-size` / `--rgb` / `--validation-split` | Folder-dataset options |
| `--no-download` | Fail instead of downloading MNIST |

---

## Checkpoint format

`checkpoints/mnist_cnn.pt` contains:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "metrics": {"train_loss", "train_accuracy", "test_loss", "test_accuracy"},
    "metadata": {"classes", "grayscale", "image_size", "history", ...},
}
```

Load it later with:

```python
import torch
from SimpleNetwork import SimpleNet

ckpt = torch.load("checkpoints/mnist_cnn.pt", map_location="cpu")
model = SimpleNet(in_channels=1, num_classes=len(ckpt["metadata"]["classes"]))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

---

## Design notes

- Standard PyTorch primitives only — no training framework on top.
- Data loading, model, training and inference live in separate modules.
- CLI configuration means no source edits to run experiments.
- Adaptive pooling lets different input sizes reach the classifier.

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the full write-up.

## License

[MIT](LICENSE).
