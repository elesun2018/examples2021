# -*- coding: utf-8 -*-
"""
   miniimagenet-cls-train-torch
fun:
    mini-imagenet分类模型训练
env:
    numpy==1.19.5 scikit-learn==0.24.2
    opencv-python==4.5.1.48 Pillow8.2.0
    tensorflow-gpu==1.14.0 Keras==2.2.5
    torch==1.4.0 torchvision==0.5.0
ref:
    https://blog.csdn.net/heiheiya/article/details/103028543 ResNet-50迁移学习进行图像分类训练
    https://blog.csdn.net/u010397980/article/details/86385383 分类及预测示例
    https://blog.csdn.net/yinxian9019/article/details/106763892 pytorch gpu
    https://blog.csdn.net/qq_33254870/article/details/103362621 ImageFolder
    https://blog.csdn.net/qq_39507748/article/details/105394808 ImageFolder
    https://blog.csdn.net/weixin_45292794/article/details/108227437 网络参数统计torchsummary
"""
import os
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
#################################GPU config#############################################
print("#"*30,"GPU config","#"*30)
print("torch version :",torch.__version__)
print("cuda version :",torch.version.cuda)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # elesun
print("GPU available :",torch.cuda.is_available())
print("GPU count :",torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda cuda:0 cuda:1 cpu
print("GPU current :",torch.cuda.current_device(),torch.cuda.get_device_name(torch.cuda.current_device()))
print("device: ", device)
models_dir = "models"
result_dir = "result"
if os.path.exists(models_dir):
    shutil.rmtree(models_dir)
os.makedirs(models_dir)
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
#################################data process#############################################
print("#"*30,"data process","#"*30)
# 数据集中train中每个类别对应一个文件夹，每个文件夹下有x张图片，val中每个类别对应一个文件夹，每个文件夹下有x张图片
data_dir = "../../../../../datasets/cls/miniimagenet"
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'val')
img_size = 256 # 256
batch_size = 32
# 数据增强
train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
valid_transforms = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)
print("len train_dataset:",len(train_dataset))
print("len valid_dataset:",len(valid_dataset))
print("train_dataset.classes:",train_dataset.classes)
print("train_dataset.class_to_idx:",train_dataset.class_to_idx)
# print("train_dataset.imgs:",train_dataset.imgs)
# 加载数据
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
#################################build model#############################################
print("#"*30,"build model","#"*30)
num_classes = 3
# 预训练模型
network = models.resnet50(pretrained=True) # resnet50 resnet101
# for param in network.parameters():
#     param.requires_grad = False # 默认为true,由于预训练的模型中的大多数参数已经训练好了，因此将requires_grad字段重置为false。

# 为了适应自己的数据集，将ResNet-50的最后一层替换为，将原来最后一个全连接层的输入喂给一个有256个输出单元的线性层，
# 接着再连接ReLU层和Dropout层，然后是256 x num_classes的线性层，输出为num_classes通道的softmax层。
fc_inputs = network.fc.in_features
network.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes), #
    nn.LogSoftmax(dim=1)
)
network = network.to(device)
# 定义损失函数和优化器
loss_function = nn.NLLLoss()
optimizer = optim.Adam(network.parameters())
# from torchstat import stat
# stat(network,(3,img_size,img_size))    # (3,224,224)表示输入图片的尺寸
from torchsummary import summary
summary(network, input_size=(3, img_size, img_size), batch_size=-1)
#################################train model#############################################
print("#"*30,"train model","#"*30)
def train_and_val(network, loss_function, optimizer, epochs=25):
    history = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        network.train()
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
        with torch.no_grad():
            network.eval()
            for j, (inputs, labels) in enumerate(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = network(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                val_acc += acc.item() * inputs.size(0)
        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = train_acc / len(train_dataset)
        avg_val_loss = val_loss / len(valid_dataset)
        avg_val_acc = val_acc / len(valid_dataset)
        history.append([avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc])
        if best_acc < avg_val_acc:
            best_acc = avg_val_acc
            best_epoch = epoch + 1
        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.3f}, Accuracy: {:.3f}, \n\t\tvalation: Loss: {:.3f}, Accuracy: {:.3f}, Time: {:.2f}s".format(
                epoch + 1, avg_val_loss, avg_train_acc, avg_val_loss, avg_val_acc,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for valation : {:.3f} at epoch {:04d}".format(best_acc, best_epoch))
        torch.save(network, os.path.join(models_dir, 'model_epoch{:04d}_valacc{:.3f}.pt'.format(epoch,avg_val_acc)))
    return network, history

num_epochs = 30
trained_model, history = train_and_val(network, loss_function, optimizer, num_epochs)
#################################visual results#############################################
print("#"*30,"visual results","#"*30)
history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(os.path.join(result_dir, 'accuracy_curve.png'))
plt.show()