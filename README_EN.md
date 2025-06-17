# 🌟 AlexNet Reproduction with PyTorch

Welcome! This repository features a reproduction of the classic **AlexNet** architecture implemented using PyTorch. The project is designed as a hands-on learning exercise and demonstrates both training from scratch and transfer learning with pretrained weights.

---

## 📚 Project Overview

- **Goal:** Reproduce and experiment with AlexNet using PyTorch.
- **Datasets:** CIFAR10 (from torchvision) and a Flowers Classification dataset (downloaded separately).
- **Features:** Custom model head for different datasets, support for transfer learning, and ready-to-use scripts for training, evaluation, and inference.

---

## 🛠️ Getting Started

### 1. Dataset Preparation

- **CIFAR10:** Downloaded automatically via `torchvision`.
- **Flowers Dataset:** Please refer to [`dataset_download.md`](dataset_download.md) for detailed instructions.

### 2. Pretrained Weights

- **Why:** Training AlexNet from scratch on small or non-ImageNet datasets often yields suboptimal results.
- **How:** To simplify workflow, pretrained weights are included in the repo and auto-downloaded via `pretrained_alexnet.py`.
- **Usage:** Just run `pretrained_alexnet.py` to set up transfer learning.

---

## 🏗️ Project Structure

```
├── model.py                # AlexNet model with dataset-adaptive head (10 classes for CIFAR10, 5 for Flowers)
├── data_preprocess.py      # Data preprocessing utilities
├── train.py                # Script to train from scratch
├── pretrained_alexnet.py   # Script for transfer learning with pretrained weights
├── test.py                 # Model evaluation and inference
├── model_test_images/      # Folder for sample test images (add your own!)
├── dataset_download.md     # Dataset download instructions
└── README.md               # You're here!
```

---

## 🚀 Usage Guide

### 1. Train a Custom AlexNet

- Edit the dataset selection and hyperparameters at the start of `train.py`.
- Run:
  ```bash
  python train.py
  ```
- All necessary folders are created automatically.

### 2. Transfer Learning

- As with training, select the dataset in `pretrained_alexnet.py`.
- Run:
  ```bash
  python pretrained_alexnet.py
  ```

### 3. Model Evaluation

After training, the `/model` directory will contain:

- `alexnet_cifar10.pth`
- `alexnet_flowers.pth`
- `alexnet_cifar10_with_pretrained.pth`
- `alexnet_flowers_with_pretrained.pth`

To test:
1. Open `test.py` and set the model and dataset at the top.
2. Specify the image path(s) for inference.
3. Run:
   ```bash
   python test.py
   ```

---

## 📈 Training Results

| Dataset      | Model Type       | Accuracy / Confidence |
|--------------|------------------|----------------------|
| **CIFAR10**  | Custom           | High (up to 100%)    |
| **CIFAR10**  | Pretrained       | **100%**             |
| **Flowers**  | Custom           | Poor (cannot classify)|
| **Flowers**  | Pretrained       | ~70% confidence      |

- **Note:** Flower dataset is challenging due to fine-grained categories and limited data; AlexNet may be too simple for this task.
- **Tip:** For best results, use transfer learning with pretrained weights.

---

## 📝 Notes & Recommendations

- Fine-tuning on small or fine-grained datasets may require more advanced architectures.
- Always verify dataset paths and adjust preprocessing to match your data.

---

## 🤝 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [AlexNet Paper (2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

---

> _Happy coding and deep learning!_ 🚀
