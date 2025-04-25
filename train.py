import torch
from model import*
from data_preprocess import get_data_cifar10, get_data_flowers
import torchvision
import os
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

def train():
    # 创建保存模型和可视化结果的目录
    os.makedirs("./model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    # 设置数据集
    # dataset = 'cifar10'  # 数据集名称
    dataset = 'flowers'  # 数据集名称


    
    # 超参数
    learning_rate = 0.001
    # 训练的轮数
    epoch = 25
    # 批次大小
    batch_size = 64  # 在获取数据函数时传入这个值
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU可用
    print(f"使用设备: {device}")

    # 初始化TensorBoard
    writer = SummaryWriter('./logs')
    
    # 获取数据集
    if dataset == 'flowers':
        train_dataloader, test_dataloader, flower_class = get_data_flowers(batch_size)
    else:
        train_dataloader, test_dataloader, classes = get_data_cifar10(batch_size)
 
    print(f"训练批次数: {len(train_dataloader)}, 测试批次数: {len(test_dataloader)}")

    # 根据数据集的种类加载不同的模型
    # 主要区别在于最后一层的输出类别数
    if dataset == 'flowers':
        model = Alexnet_Flowers()
    else:
        model = Alexnet_Cifar10()

    model = model.to(device)  # 将模型放到GPU上
    
    # 记录模型结构到TensorBoard
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)  # 将损失函数放到GPU上

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=1e-4)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 用于存储训练和测试指标的列表
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    # 记录最佳测试准确率和对应的epoch
    best_acc = 0.0
    best_epoch = 0
    
    # 记录训练开始时间
    start_time = time.time()

    # 训练和测试循环
    for i in range(epoch):
        epoch_start_time = time.time()
        print(f"\n=======第{i + 1}/{epoch}轮训练开始=======")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        
        # 使用tqdm显示进度条
        train_bar = tqdm(train_dataloader, desc=f"Training Epoch {i+1}")
        for data in train_bar:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            
            # 前向传播
            output = model(imgs)
            loss = loss_fn(output, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred = output.argmax(dim=1)
            correct = (pred == targets).sum().item()
            total = targets.size(0)
            
            # 累计损失和准确率
            train_loss += loss.item() * imgs.size(0)
            train_acc += correct
            train_samples += total
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        # 计算整个epoch的平均损失和准确率
        avg_train_loss = train_loss / train_samples
        avg_train_acc = train_acc / train_samples
        train_losses.append(avg_train_loss)
        
        # 添加到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, i)
        writer.add_scalar('Accuracy/train', avg_train_acc, i)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_samples = 0
        
        with torch.no_grad():
            test_bar = tqdm(test_dataloader, desc=f"Testing Epoch {i+1}")
            for data in test_bar:
                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)
                
                # 前向传播
                output = model(imgs)
                loss = loss_fn(output, targets)
                
                # 计算准确率
                pred = output.argmax(dim=1)
                correct = (pred == targets).sum().item()
                total = targets.size(0)
                
                # 累计损失和准确率
                test_loss += loss.item() * imgs.size(0)
                test_acc += correct
                test_samples += total
                
                # 更新进度条
                test_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct/total:.4f}"
                })
        
        # 计算整个测试集的平均损失和准确率
        avg_test_loss = test_loss / test_samples
        avg_test_acc = test_acc / test_samples
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_test_acc)
        
        # 添加到TensorBoard
        writer.add_scalar('Loss/test', avg_test_loss, i)
        writer.add_scalar('Accuracy/test', avg_test_acc, i)
        
        # 更新学习率调度器
        scheduler.step(avg_test_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, i)
        
        # 计算当前epoch花费的时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印当前epoch的结果
        print(f"Epoch {i+1}/{epoch} - Time: {epoch_time:.2f}s - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        
        # # 保存最佳模型
        # if avg_test_acc > best_acc:
        #     best_acc = avg_test_acc
        #     best_epoch = i + 1
        #     torch.save(model.state_dict(), "./model/alexnet_best.pth")
        #     print(f"保存最佳模型 (Acc: {best_acc:.4f})")
    
    # 保存最终模型参数
    if dataset == 'flowers':
        torch.save(model.state_dict(), "./model/alexnet_flowers.pth")
    else:
        torch.save(model.state_dict(), "./model/alexnet_cifar10.pth")
    print("保存最终模型成功！")
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n训练完成! 总时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"最佳测试准确率: {best_acc:.4f} (Epoch {best_epoch})")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epoch+1), test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch+1), test_accuracies, label='Test Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./plots/training_results.png')
    plt.close()
    
    print(f"训练结果图表已保存到 './plots/training_results.png'")
    print("模型保存成功！")

if __name__ == '__main__':
    train()