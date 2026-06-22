from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn

from SimpleNetwork import SimpleNet
from data_load import build_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an image classifier.")
    parser.add_argument("--dataset", choices=["mnist", "imagefolder"], default="mnist", help="Dataset type to train on.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Force device, for example cpu or cuda.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional training subset size for quick experiments.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional test subset size for quick experiments.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mnist_cnn.pt"), help="Checkpoint path.")
    parser.add_argument("--no-download", action="store_true", help="Fail instead of downloading MNIST if missing.")
    parser.add_argument("--image-size", type=int, default=28, help="Image size used for image-folder datasets.")
    parser.add_argument("--rgb", action="store_true", help="Use RGB input for image-folder datasets instead of grayscale.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split used for image-folder datasets.")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None):
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits, targets):
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct, targets.size(0)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        batch_correct, batch_total = accuracy_from_logits(logits, targets)
        correct += batch_correct
        total += batch_total

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            batch_correct, batch_total = accuracy_from_logits(logits, targets)
            correct += batch_correct
            total += batch_total

    return running_loss / total, correct / total


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_loader, test_loader, metadata = build_loaders(
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        data_dir=args.data_dir,
        download=not args.no_download,
        dataset_type=args.dataset,
        image_size=args.image_size,
        grayscale=not args.rgb,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    in_channels = 1 if metadata.get("grayscale", True) else 3
    num_classes = len(metadata["classes"])

    model = SimpleNet(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        history.append(metrics)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

    duration = perf_counter() - start
    final_metrics = history[-1]
    metadata = {
        **metadata,
        "dataset_type": args.dataset,
        "device": str(device),
        "history": history,
        "image_size": args.image_size,
    }
    save_checkpoint(args.output, model, optimizer, args.epochs, final_metrics, metadata)

    print(f"Saved checkpoint to {args.output}")
    print(f"Total time: {duration:.1f}s")
    print(
        "Final metrics: "
        f"train acc {final_metrics['train_accuracy']:.4f}, "
        f"test acc {final_metrics['test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
