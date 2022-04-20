# -*- coding: utf-8 -*-
"""
   miniimagenet-cls-eval-torch
fun:
    mini-imagenet分类模型评估
env:
    numpy==1.19.5 scikit-learn==0.24.2
    opencv-python==4.5.1.48 Pillow8.2.0
    tensorflow-gpu==1.14.0 Keras==2.2.5
    torch==1.4.0 torchvision==0.5.0
ref:

    https://blog.csdn.net/u010397980/article/details/86385383 分类及预测示例
    https://blog.csdn.net/yinxian9019/article/details/106763892 pytorch gpu
    https://blog.csdn.net/weixin_45885074/article/details/114065326 基于Pytorch的模型推理
    https://blog.csdn.net/heiheiya/article/details/103028543 ResNet-50迁移学习进行图像分类训练
    https://goodgoodstudy.blog.csdn.net/article/details/104306781 pytorch 迁移学习多分类
    https://blog.csdn.net/heiheiya/article/details/103031300 训练好的模型进行图像分类预测
"""
import os
import numpy as np
import torch
from torchvision import datasets, models, transforms
import time
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, precision_recall_curve, average_precision_score,auc

def model_valid(valid_dir, model_path):
    #################################data process#############################################
    print("#" * 30, "data process", "#" * 30)
    # 数据集中train中每个类别对应一个文件夹，每个文件夹下有x张图片，val中每个类别对应一个文件夹，每个文件夹下有x张图片
    img_size = 256  # 256
    batch_size = 32
    # 数据增强
    valid_transforms = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)
    print("len valid_dataset:", len(valid_dataset))
    print("train_dataset.classes:", valid_dataset.classes)
    print("train_dataset.class_to_idx:", valid_dataset.class_to_idx)
    # print("train_dataset.imgs:",valid_dataset.imgs)
    # 加载数据
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    #################################load model#############################################
    print("#" * 30, "load model", "#" * 30)
    # num_classes = 3
    model = torch.load(model_path)
    # 预训练模型
    # network = models.resnet101(pretrained=False) # resnet50 resnet101
    # network.load_state_dict(model)#加载模型
    model = model.to(device)
    model.eval()
    #################################model valation#############################################
    print("#" * 30, "model valation", "#" * 30)
    epoch_start = time.time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_dataloader):
            lab_np = labels.numpy()
            # print("lab_np.shape:",lab_np.shape)
            inputs = inputs.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)
            ret, predict = torch.max(outputs.data, 1)
            pred_np = predict.cpu().numpy()
            # print("pred_np.shape:",pred_np.shape)
            if i == 0 :
                gt_np = lab_np
                pd_np = pred_np
            else :
                gt_np = np.append(gt_np,lab_np)
                # print("gt_np:\n", gt_np)
                pd_np = np.append(pd_np, pred_np)
                # print("pd_np:\n", pd_np)
    epoch_end = time.time()
    print("gt_np.shape: {}, pd_np.shape: {}, time use: {:.3f}s".format(gt_np.shape, gt_np.shape, (epoch_end - epoch_start)))
    return gt_np,pd_np

def eval_metric(valid_dir, model_path):
    y_valid,y_predict = model_valid(valid_dir, model_path)
    #################################eval metrics#############################################
    print("#" * 30, "eval metrics", "#" * 30)
    # 指标评估
    print("confusion_matrix\n", confusion_matrix(y_valid, y_predict))  #
    # 准确率
    metric_accuracy = accuracy_score(y_valid, y_predict)
    print("metric_accuracy", round(metric_accuracy, 3))
    #  精准率
    metric_precision = precision_score(y_valid, y_predict, average='macro') # average='binary' [None, 'micro', 'macro', 'weighted']
    print("metric_precision", round(metric_precision, 3))
    # 召回率
    metric_recall = recall_score(y_valid, y_predict, average='macro')
    print("metric_recall", round(metric_recall, 3))
    # F1(精准率与召回率的平衡)
    metric_f1 = f1_score(y_valid, y_predict, average='macro')
    print("metric_f1", round(metric_f1, 3))

    # metric_roc = roc_auc_score(y_valid, y_predict)
    # print("metric_roc", round(metric_roc, 3))

if __name__ == "__main__":
    #################################GPU config#############################################
    print("#" * 30, "GPU config", "#" * 30)
    print("torch version :", torch.__version__)
    print("cuda version :", torch.version.cuda)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # elesun
    print("GPU available :", torch.cuda.is_available())
    print("GPU count :", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda cuda:0 cuda:1 cpu
    print("GPU current :", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    print("device: ", device)

    model_path = "models50/model_epoch0022_valacc0.936.pt"
    valid_dir = "../../../../../datasets/cls/miniimagenet/val" #
    eval_metric(valid_dir, model_path)
