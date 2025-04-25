import torchvision
import torch
import PIL.Image 
from torch import classes, nn
from model import*
import os


# 数据集类型
dataset = 'flowers'  # 'flowers' or 'cifar10'
# dataset = 'cifar10'  # 'flowers' or 'cifar10'

# 模型类型：微调的模型/自己训练的模型
model_type = "pretrained"  # 微调模型
# model_type = "train"  # 自己训练的模型

# 测试图片路径
image_path = "./model_test_image/flowers/tulips/tulip1.jpg"  # 花数据集
# image_path = "./model_test_image/cifar10/plane/plane1.png"  # cifar10数据集
if not os.path.exists(image_path):
    print(f"图片文件不存在: {image_path}")
    exit(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU可用


# 定义类别名称，用于输出预测结果
flower_class = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
cifar10_class = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 选择数据集的类别名称
if dataset == 'flowers':
    classes = flower_class  # 选择数据集的类别名称
else:
    classes = cifar10_class  # 选择数据集的类别名称



true_class_name = image_path.split('/')[3]  # 获取真实类别名称
print(f"真实类别: {true_class_name}")

image = PIL.Image.open(image_path)
image = image.convert('RGB')  # 转换为RGB格式

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)  


if model_type == "train":
    if dataset == 'flowers':
        model = Alexnet_Flowers()  # 创建模型实例
    else:
        model = Alexnet_Cifar10() # 创建模型实例
else:
    alexnet_true = torchvision.models.alexnet(pretrained = False)  # 加载预训练的AlexNet模型
    if dataset == "flowers":
        alexnet_true.classifier.add_module("added_linear",nn.Linear(1000,5))  # 如果是flowers数据集，最后一层改为5
        model = alexnet_true
    else:
        alexnet_true.classifier.add_module("added_linear",nn.Linear(1000,10)) # 如果是cifar10数据集，最后一层改为10
        model = alexnet_true

# 加载的模型参数路径
if model_type == "pretrained":
    if dataset == "flowers":
        model_path = "./model/alexnet_flowers_with_pretrained.pth"  # 模型参数路径
    else:
        model_path = "./model/alexnet_cifar10_with_pretrained.pth"  # 模型参数路径
else:
    if dataset == "flowers":
        model_path = "./model/alexnet_flowers.pth"
    else:
        model_path = "./model/alexnet_cifar10.pth"


# 判断模型参数路径是否存在
if not os.path.exists(model_path):
    print(f"模型参数文件不存在: {model_path}")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型参数
# print(model)  

model = model.to(device)  # 将模型放到GPU上
model.eval()  # 设置模型为评估模式

image = torch.reshape(image, (1, 3, 224, 224))  # 将图片的维度调整为(1, 3, 32, 32)，batchsize为1，通道数为3，宽高为32*32
image = image.to(device)  # 将数据放到GPU上

with torch.no_grad():  # 不计算梯度,可以节约内存
    output = model(image)
print(output)
print(output.argmax(1))  # 输出预测的类别

# 计算softmax获取概率
probabilities = torch.nn.functional.softmax(output, dim=1)
pred_idx = output.argmax(1).item()  # 获取预测的类别索引
pred_class = classes[pred_idx]  # 获取预测的类别名称
pred_prob = probabilities[0][pred_idx].item()  # 获取预测的概率值

# 输出预测结果
print(f"\n模型路径: {model_path}")
print(f"\n预测结果")
print(f"真实类别: {true_class_name}")
print(f"预测类别: {pred_class}")

print("\n所有类别预测概率")
for i, cls in enumerate(classes):
    print(f"{cls}: {probabilities[0][i].item()*100:.2f}%")

# 判断预测是否正确
if true_class_name in classes:
    true_idx = classes.index(true_class_name)
    is_correct = pred_idx == true_idx
    print(f"\n预测结果: {'✓ 正确' if is_correct else '✗ 错误'}")

