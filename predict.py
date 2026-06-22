from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from SimpleNetwork import SimpleNet
from data_load import build_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class for a single image.")
    parser.add_argument("image", type=Path, help="Path to the image to classify.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mnist_cnn.pt"), help="Trained checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="Force device, for example cpu or cuda.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to print.")
    return parser.parse_args()


def resolve_device(device_name: str | None):
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    classes = metadata.get("classes", [str(i) for i in range(10)])
    grayscale = metadata.get("grayscale", True)
    image_size = metadata.get("image_size", 28)
    in_channels = 1 if grayscale else 3

    model = SimpleNet(in_channels=in_channels, num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, grayscale, image_size


def preprocess_image(image_path: Path, grayscale: bool, image_size: int):
    image = Image.open(image_path)
    transform = build_transform(image_size=image_size, grayscale=grayscale)
    return transform(image).unsqueeze(0)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, classes, grayscale, image_size = load_checkpoint(args.checkpoint, device)

    image_tensor = preprocess_image(args.image, grayscale=grayscale, image_size=image_size).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_k = min(args.top_k, probabilities.numel())
        values, indices = torch.topk(probabilities, k=top_k)

    best_index = indices[0].item()
    best_class = classes[best_index]
    best_probability = values[0].item()

    print(f"Prediction: {best_class}")
    print(f"Confidence: {best_probability:.4f}")
    print("Top predictions:")
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"  {classes[index]}: {value:.4f}")


if __name__ == "__main__":
    main()
