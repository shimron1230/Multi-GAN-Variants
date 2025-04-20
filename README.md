# 🔥 Multi-GAN Framework (Original, Softmax, Arctan, LS)

This project implements a flexible GAN training pipeline in PyTorch that supports multiple GAN variants and datasets. It provides options to experiment with different loss functions and generator-discriminator objectives.

## 📦 Supported GAN Types

| Type     | Description |
|----------|-------------|
| `origin` | Standard GAN with binary cross-entropy loss |
| `softmax` | Softmax GAN using cross-entropy loss in batch sample space |
| `arctan` | Uses a normalized arctangent function instead of softmax |
| `ls`     | Least Squares GAN using MSE loss with softmax-style probabilities |

## 📊 Supported Datasets

- `mnist`
- `fashion-mnist`
- `celeba` (images are resized to 64×64)

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision numpy
```

### 2. Run Training

```bash
python main.py --gan_type softmax --dataset mnist --n_epochs 200
```

### 3. Available Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--gan_type` | Type of GAN (`origin`, `softmax`, `arctan`, `ls`) | `origin` |
| `--dataset` | Dataset to train on (`mnist`, `fashion-mnist`, `celeba`) | `mnist` |
| `--n_epochs` | Number of training epochs | `200` (use `50` for CelebA) |
| `--batch_size` | Size of training batches | `64` |
| `--lr` | Learning rate for Adam optimizer | `0.0002` |
| `--latent_dim` | Size of latent vector (input to generator) | `100` |
| `--img_size` | Size of input/output images | `28` (use `64` for CelebA) |
| `--channels` | Number of image channels | `1` (use `3` for CelebA) |
| `--sample_interval` | Interval (in batches) to save sample images | `400` |

## 📁 Checkpoints & Outputs

- Model checkpoints are saved in `checkpoints_<gan_type>_<dataset>/`
- Generated images are saved in `images_<gan_type>_<dataset>/` every `sample_interval` batches

## 💡 GAN Logic Overview

Each GAN type has a specific loss formulation:

- **Origin GAN**: Uses BCE loss for discriminator and generator.
- **Softmax GAN**: Uses softmax cross-entropy over a batch to maintain non-zero gradients.
- **Arctan GAN**: Similar to softmax GAN but with a normalized arctangent probability function.
- **LS GAN**: Uses MSE loss instead of cross-entropy, simulating a least-squares objective.

## 📷 Sample Output
Images are saved during training at intervals for visual inspection.

## ✍️ Author

Developed by Shimron Blum  
Built on top of: https://github.com/eriklindernoren/PyTorch-GAN

---

## 📌 Notes

- For CelebA, use `--img_size 64 --channels 3 --n_epochs 50`
- Be aware that `ls` GAN can become unstable on CelebA due to infinite values in the partition function — consider using value clipping.
