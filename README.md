# 🌟 使用 PyTorch 复现 AlexNet

欢迎来到本项目！本仓库展示了如何使用 PyTorch 复现经典的 **AlexNet** 网络结构，涵盖了从零训练和迁移学习两种模式，适合深度学习入门和进阶学习者实践参考。

---

## 📚 项目简介

- **目标**：使用 PyTorch 完整复现 AlexNet，并进行实验。
- **数据集**：包含 torchvision 自带的 CIFAR10，以及需手动下载的花卉分类数据集。
- **特色**：根据不同数据集自适应调整模型分类头，兼容迁移学习，脚本即开即用，支持训练、测试和推理。

---

## 🛠️ 快速开始

### 1. 数据集准备

- **CIFAR10**：由 `torchvision` 自动下载，无需手动操作。
- **花卉数据集**：请参考 [`dataset_download.md`](dataset_download.md) 获得详细下载与整理方法。

### 2. 预训练模型

- **说明**：在小数据集或非 ImageNet 上从零训练 AlexNet 效果通常不佳，推荐使用预训练权重微调。
- **方法**：预训练参数已集成于仓库，运行 `pretrained_alexnet.py` 即可自动下载并完成迁移学习准备。
- **用法**：只需执行该脚本，迁移学习流程将自动完成。

---

## 🏗️ 项目结构

```
├── model.py                # AlexNet 主体结构，分类头可适配（CIFAR10 为10类，花卉为5类）
├── data_preprocess.py      # 数据预处理工具
├── train.py                # 从零训练脚本
├── pretrained_alexnet.py   # 迁移学习训练脚本
├── test.py                 # 模型验证与推理脚本
├── model_test_images/      # 测试图片目录（可自行添加图片）
├── dataset_download.md     # 数据集下载说明
└── README.md               # 项目说明文档
```

---

## 🚀 使用指南

### 1. 从零训练自定义 AlexNet

- 在 `train.py` 文件开头选择数据集及调整超参数（如学习率、训练轮数等）。
- 执行命令：
  ```bash
  python train.py
  ```
- 所需的文件夹会自动创建。

### 2. 迁移学习训练

- 在 `pretrained_alexnet.py` 文件开头选择数据集。
- 执行命令：
  ```bash
  python pretrained_alexnet.py
  ```

### 3. 模型验证与推理

训练或迁移学习后，`/model` 目录下会生成如下模型文件：

- `alexnet_cifar10.pth`
- `alexnet_flowers.pth`
- `alexnet_cifar10_with_pretrained.pth`
- `alexnet_flowers_with_pretrained.pth`

验证步骤：
1. 打开 `test.py`，在文件开头选择模型和数据集。
2. 指定测试图片路径或文件夹。
3. 执行命令：
   ```bash
   python test.py
   ```

---

## 📈 训练效果

| 数据集      | 模型类型           | 准确率 / 置信度         |
|-------------|--------------------|------------------------|
| **CIFAR10** | 自定义模型         | 很高（最高可达 100%）   |
| **CIFAR10** | 预训练模型         | **100%**               |
| **花卉**    | 自定义模型         | 很差（几乎无法分类）    |
| **花卉**    | 预训练模型         | 约 70% 置信度          |

- **说明**：花卉数据集为细粒度分类任务且数据量较小，AlexNet 模型结构相对简单，效果有限。建议迁移学习方式获得更优结果。

---

## 📝 注意事项

- 训练细粒度或小样本数据时，建议采用更深或更精细的网络结构。
- 请确保数据集路径正确，并根据实际数据调整预处理参数。

---

## 🤝 致谢

- [PyTorch 中文网](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [AlexNet 原论文 (2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

---

> _祝你学习愉快，深度学习之路一帆风顺！_ 🚀
