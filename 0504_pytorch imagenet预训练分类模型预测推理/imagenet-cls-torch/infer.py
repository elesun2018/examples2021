# -*- coding: utf-8 -*-
"""
   infer_cls_torch_imagenet
fun:
    imagenet预训练分类模型预测推理
env:
    numpy==1.19.5 scikit-learn==0.24.2
    opencv-python==4.5.1.48 Pillow8.2.0
    tensorflow-gpu==1.14.0 Keras==2.2.5
    torch==1.4.0 torchvision==0.5.0
ref:
    https://www.cnblogs.com/vvzhang/p/14116632.html
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # elesun
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

img = Image.open("img/n03887697_5559.JPEG")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
print("img_t.size",img_t.size()) # torch.Size([3, 224, 224])
print("batch_t.size",batch_t.size()) # torch.Size([1, 3, 224, 224])
resnet = models.alexnet(pretrained=True) # resnet101 resnet50 alexnet

# 模型推断
resnet.eval()
out = resnet(batch_t)
print("out.size",out.size()) # torch.Size([1, 1000])

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0]
print("label : ",classes[index[0]])
print("proba : ",percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
top5_list = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
print("top5_list : \n",top5_list)


