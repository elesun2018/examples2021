# -*- coding: utf-8 -*-
"""
   miniimagenet-cls-infer-torch
fun:
    mini-imagenet分类模型预测推理
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
import torch
from PIL import Image
from torchvision import datasets, models, transforms
import time
#################################GPU config#############################################
print("#"*30,"GPU config","#"*30)
print("torch version :",torch.__version__)
print("cuda version :",torch.version.cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # elesun
print("GPU available :",torch.cuda.is_available())
print("GPU count :",torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda cuda:0 cuda:1 cpu
print("GPU current :",torch.cuda.current_device(),torch.cuda.get_device_name(torch.cuda.current_device()))
print("device: ", device)
#################################data process#############################################
print("#"*30,"data process","#"*30)
# 数据集中train中每个类别对应一个文件夹，每个文件夹下有x张图片，val中每个类别对应一个文件夹，每个文件夹下有x张图片
data_dir = "../../../../../datasets/cls/miniimagenet"
test_dir = os.path.join(data_dir, 'test')
img_size = 256 # 256
batch_size = 32
label_dict = {
    "n02123045" : 0,
    "n03887697" : 1,
    "n03888257" : 2
}
# 数据增强
test_transforms = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
#################################load model#############################################
print("#"*30,"load model","#"*30)
models_path = "models50/model_epoch0022_valacc0.936.pt"
# num_classes = 3
model = torch.load(models_path)
# 预训练模型
# network = models.resnet101(pretrained=False) # resnet50 resnet101
# network.load_state_dict(model)#加载模型
model = model.to(device)
#################################model infer#############################################
print("#"*30,"model infer","#"*30)
def model_infer(model, test_dir, test_transforms):
    predict_list = []
    torch.no_grad()
    model.eval()
    img_list = os.listdir(test_dir)
    print("len test_dataset:", len(img_list))
    for img_name in img_list:
        img_path = os.path.join(test_dir,img_name)
        img = Image.open(img_path)
        print("predicting for image : ",img_path)
        epoch_start = time.time()
        inputs = test_transforms(img).unsqueeze(0)  # 由于训练的时候还有一个参数，是batch_size,而推理的时候没有，所以我们为了保持维度统一，就得使用.unsqueeze(0)来拓展维度
        inputs = inputs.to(device)  # 同样将图片数据放入cuda(GPU)中
        outputs = model(inputs)
        ret, predict = torch.max(outputs, 1)
        epoch_end = time.time()
        pred = predict.cpu().numpy()[0]
        predict_list.append(pred)
        for key in label_dict.keys():
            if label_dict[key] == pred : label=key;break
        print("Img_name:{}\t, Pred_val:{}\t, Label_name:{}\t, Time_use: {:.3f}s\n\n".format(
            img_name, pred, label, epoch_end - epoch_start))
    return predict_list

y_pred = model_infer(model, test_dir, test_transforms)
print("y_pred",y_pred)

