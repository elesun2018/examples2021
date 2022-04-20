# -*- coding: utf-8 -*-
'''
    计算PyTorch模型运算量flops和参数量
fun:
    计算PyTorch模型运算量flops和参数量
ref:
    https://blog.csdn.net/weixin_41519463/article/details/102468868
    https://www.jb51.net/article/177487.htm
    https://blog.csdn.net/weixin_43519707/article/details/108512769
'''
import torch
from torchvision.models import *
from thop import profile
model = resnet101() # mobilenet_v2 vgg16 resnet18 resnet101 densenet201
input = torch.randn(1, 3, 224, 224) #模型输入的形状,batch_size=1
flops, params = profile(model, inputs=(input, ))
print("flops:%.3fG,params:%.3fM"%(flops/1e9,params/1e6)) #flops单位G，para单位M