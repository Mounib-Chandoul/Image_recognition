from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
DEFAULT_IMAGE_SIZE = 28


def build_transform(image_size: int = DEFAULT_IMAGE_SIZE, grayscale: bool = True) -> transforms.Compose:
    transform_steps = []
    if grayscale:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    transform_steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    if grayscale:
        normalize = (0.5,)
        transform_steps.append(transforms.Normalize(normalize, normalize))
    else:
        transform_steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_steps)


def load_mnist_datasets(data_dir: Path | str = DATA_DIR, download: bool = True):
    transform = build_transform(image_size=28, grayscale=True)
    root = str(data_dir)
    train_set = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    return train_set, test_set


def load_imagefolder_datasets(
    data_dir: Path | str,
    image_size: int = 28,
    grayscale: bool = True,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Sequence[str]]:
    root = Path(data_dir)
    dataset_root = root / "train" if (root / "train").exists() else root
    dataset = datasets.ImageFolder(root=str(dataset_root), transform=build_transform(image_size=image_size, grayscale=grayscale))

    if len(dataset) == 0:
        raise ValueError(f"No images found under {dataset_root}")

    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")

    val_size = max(1, int(len(dataset) * validation_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("dataset is too small for the requested validation split")

    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_size, val_size], generator=generator)
    return train_set, test_set, dataset.classes


def _maybe_subset(dataset, limit: Optional[int]):
    if limit is None:
        return dataset
    limit = max(0, min(limit, len(dataset)))
    return Subset(dataset, range(limit))


def build_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    data_dir: Path | str = DATA_DIR,
    download: bool = True,
    dataset_type: str = "mnist",
    image_size: int = DEFAULT_IMAGE_SIZE,
    grayscale: bool = True,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, dict]:
    if dataset_type == "mnist":
        train_set, test_set = load_mnist_datasets(data_dir=data_dir, download=download)
        metadata = {"dataset_type": "mnist", "classes": [str(i) for i in range(10)], "image_size": 28, "grayscale": True}
    elif dataset_type == "imagefolder":
        train_set, test_set, classes = load_imagefolder_datasets(
            data_dir=data_dir,
            image_size=image_size,
            grayscale=grayscale,
            validation_split=validation_split,
            seed=seed,
        )
        metadata = {
            "dataset_type": "imagefolder",
            "classes": list(classes),
            "image_size": image_size,
            "grayscale": grayscale,
        }
    else:
        raise ValueError("dataset_type must be either 'mnist' or 'imagefolder'")

    train_set = _maybe_subset(train_set, train_limit)
    test_set = _maybe_subset(test_set, test_limit)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader, metadata
