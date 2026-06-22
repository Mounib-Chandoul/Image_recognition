# Image Recognition Project Report

## What this project now does
This repository is now a complete image-classification project built with PyTorch.

It can:
- download and load the MNIST handwritten digit dataset
- preprocess images into normalized tensors
- define a convolutional neural network for classification
- train the model end-to-end
- evaluate the model on the test set after each epoch
- save a checkpoint containing the trained weights and optimizer state
- train on a custom folder-based image dataset
- run inference on a single image from a saved checkpoint

## File-by-file breakdown

### `data_load.py`
This module is responsible for dataset and dataloader creation.

What it provides:
- `build_transform()` creates the preprocessing pipeline
- `load_mnist_datasets()` loads MNIST train and test splits
- `load_imagefolder_datasets()` builds train/test splits from an `ImageFolder` layout
- `build_loaders()` builds dataloaders and returns dataset metadata
- optional `train_limit` and `test_limit` arguments let you run quick experiments on smaller subsets

Why this matters:
- it keeps data handling out of the training script
- it makes it easy to reuse the same loaders from other scripts later
- it gives you one place to change preprocessing if you switch datasets

### `SimpleNetwork.py`
This module defines the model architecture.

Current architecture:
- two convolution layers
- max pooling after each convolution
- dropout regularization
- one hidden fully connected layer
- adaptive pooling so different input image sizes can still flow into the classifier
- final classifier sized to the number of classes in the dataset

Why this is better than the original version:
- the old network had only one convolution layer and went straight to the classifier
- the current version has more feature extraction capacity and supports more than one dataset shape

### `main.py`
This is the training and evaluation entry point.

It now handles:
- command-line arguments for training configuration
- reproducible seeding
- device selection for CPU or CUDA
- one-epoch training loop
- test-set evaluation loop
- checkpoint saving to `checkpoints/mnist_cnn.pt`
- MNIST or custom `ImageFolder` training through `--dataset`
- grayscale or RGB folder inputs through `--rgb`

Main CLI options:
- `--dataset`: `mnist` or `imagefolder`
- `--data-dir`: dataset root path
- `--epochs`: number of epochs to train
- `--batch-size`: batch size for loaders
- `--learning-rate`: Adam learning rate
- `--device`: force `cpu` or `cuda`
- `--train-limit` and `--test-limit`: optional smaller subsets for quick runs
- `--output`: checkpoint path
- `--no-download`: stop if MNIST is not already present locally
- `--image-size`: resize used for folder datasets
- `--rgb`: keep folder images in RGB instead of grayscale
- `--validation-split`: train/validation split ratio for folder datasets

### `predict.py`
This script loads a saved checkpoint and predicts the class for one image.

What it does:
- loads the saved model weights
- restores the class list from checkpoint metadata when available
- applies the same preprocessing pipeline used for training
- prints the top predictions and probabilities

## How the training works
The training loop does this each epoch:
1. switch the model into training mode
2. iterate over the training loader
3. move inputs and labels to the selected device
4. compute logits
5. compute cross-entropy loss
6. backpropagate gradients
7. update model weights with Adam
8. compute training accuracy
9. run a full evaluation pass on the test set
10. print both training and test metrics

## Output artifacts
After training, the script writes a checkpoint file containing:
- current epoch
- model parameters
- optimizer state
- final metrics
- full metric history
- dataset metadata
- device information

The default path is:
- `checkpoints/mnist_cnn.pt`

## How to run it
Typical full training run:
```bash
python main.py
```

Quick development run on smaller subsets:
```bash
python main.py --epochs 2 --train-limit 5000 --test-limit 1000
```

Force CPU:
```bash
python main.py --device cpu
```

Change batch size and learning rate:
```bash
python main.py --batch-size 128 --learning-rate 0.0005
```

Train on a custom folder dataset:
```bash
python main.py --dataset imagefolder --data-dir path/to/dataset --epochs 10
```

Folder dataset layout expected by `ImageFolder`:
```text
path/to/dataset/
  train/
    class_a/
      image1.jpg
      image2.jpg
    class_b/
      ...
```

Run a single-image prediction:
```bash
python predict.py path/to/image.jpg --checkpoint checkpoints/mnist_cnn.pt
```

## How to use the trained model later
The checkpoint can be loaded with `torch.load`, then the saved `model_state_dict` can be restored into `SimpleNet()`.

Typical pattern:
```python
import torch
from SimpleNetwork import SimpleNet

checkpoint = torch.load("checkpoints/mnist_cnn.pt", map_location="cpu")
model = SimpleNet()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

If the checkpoint was trained on an image-folder dataset, the metadata also stores:
- class names
- whether the model expects grayscale or RGB inputs
- the image size used during training

## Suggested next improvements
If you want to extend the project yourself, the most useful upgrades are:
- add a small batch prediction script for whole directories
- add accuracy/loss plotting after training
- add early stopping and best-checkpoint selection
- switch from a simple resize pipeline to augmentation for harder datasets
- save the best checkpoint separately from the final checkpoint

## Design choices made here
- Kept the project simple enough to understand and modify without a framework on top
- Used standard PyTorch primitives only
- Kept the code split into data, model, training, and inference concerns
- Added command-line configuration so the scripts are usable without editing source every time

## Current limitation
This project is still intentionally lightweight. If you want to train on more complex image domains, you will likely need to:
- tune the network depth
- tune the image size
- use augmentation
- add a larger model backbone
