# Reproduce Alexnet using PyTorch
## 1. 项目描述
本仓库包含我在学习PyTorch框架时实现的AlexNet网络。

## 2. 准备工作
### 2.1 数据集下载
详细下载说明请参阅`dataset_download.md`文件  
本实验使用：
- torchvision中的CIFAR10数据集
- 从网上下载的花卉分类数据集

### 2.2 预训练模型
关于预训练模型的说明：  
在使用自有数据集训练时，由于数据集质量问题（原计划使用ImageNet但计算资源不足），模型表现欠佳。  
因此从torchvision下载了预训练的AlexNet参数。

为方便本地使用，参数文件保存在仓库中。为简化GitHub下载流程，我将下载任务集成在`pretrain_alexnet.py`中 - 用户只需运行该文件即可进行迁移学习。

## 3. 项目结构
- `model.py`：自定义模型文件，针对不同数据集配置最终线性层：
  - CIFAR10（10分类）
  - 花卉数据集（5分类）
- `data_preprocess.py`：数据预处理
- `train.py`：使用自定义网络训练
- `pretrained_alexnet.py`：使用预训练模型进行迁移学习
- `test.py`：模型验证
- `model_test_images`：测试图片（用户可自行添加）

## 4. 使用说明
### 4.1 自定义网络训练
直接运行`train.py`会自动创建所需文件夹。  
除学习率、训练轮数等超参数外，用户需在文件开头选择使用的数据集。

### 4.2 迁移学习
与`train.py`类似 - 在开头选择数据集后运行即可。

### 4.3 模型验证
完成4.2步骤后，`/model`目录将生成四个模型文件：
- `alexnet_cifar10.pth`
- `alexnet_flowers.pth` 
- `alexnet_cifar10_with_pretrained.pth`
- `alexnet_flowers_with_pretrained.pth`

使用`test.py`进行验证：
1. 选择模型类型和数据集（代码开头已预定义）
2. 指定测试图片路径

## 5. 训练结果
### 5.1 CIFAR10数据集
自定义模型和预训练模型均表现优异，其中使用预训练参数达到了100%准确率。

### 5.2 花卉数据集
性能较差，可能原因：
- AlexNet结构相对简单，难以处理花卉这种细粒度分类任务
- 自定义模型完全无法正确识别
- 预训练模型虽能识别但置信度仅约70%
