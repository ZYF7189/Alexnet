# AlexNet
## 1. Description
This repository contains my PyTorch implementation of AlexNet while learning the framework.

## 2. Setup
### 2.1 Dataset Download
Refer to `dataset_download.md` for detailed download instructions.  
This experiment uses:
- CIFAR10 dataset from torchvision
- Flower classification dataset downloaded from the web

### 2.2 Pretrained Models
Notes about pretrained models:  
When training with my datasets, the model performance was suboptimal due to dataset quality issues (originally wanted ImageNet but lacked sufficient computing resources).  
Therefore, I downloaded pretrained AlexNet parameters from torchvision.  

For local convenience, parameter files are stored in the repository. To simplify GitHub downloads, I included the download task in `pretrain_alexnet.py` - users only need to run this file for transfer learning.

## 3. Structure
- `model.py`: Custom model file with different final linear layer configurations for:
  - CIFAR10 (10 classes)
  - Flowers dataset (5 classes)
- `data_preprocess.py`: Data preprocessing
- `train.py`: Training with custom network
- `pretrained_alexnet.py`: Transfer learning with pretrained model
- `test.py`: Model validation
- `model_test_images`: Test images (users can add their own)

## 4. Usage
### 4.1 Custom Network Training
Run `train.py` directly to create necessary folders.  
Besides hyperparameters (learning rate, epochs, etc.), users need to select the dataset at the beginning of the file.

### 4.2 Transfer Learning
Same as `train.py` - select dataset at the beginning and run.

### 4.3 Model Validation
After completing 4.2, four models will appear in `/model`:
- `alexnet_cifar10.pth` 
- `alexnet_flowers.pth`
- `alexnet_cifar10_with_pretrained.pth`  
- `alexnet_flowers_with_pretrained.pth`

Use `test.py` for validation:
1. Select model type and dataset (predefined in code)
2. Specify test image path

## 5. Training Results
### 5.1 CIFAR10 Dataset
Both custom and pretrained models performed exceptionally well, with pretrained parameters achieving 100% accuracy.

### 5.2 Flowers Dataset
Performance was poorer, likely because:
- AlexNet's architecture is too simple for fine-grained flower classification
- Custom models failed completely in recognition
- Pretrained models succeeded but with only ~70% confidence probability
