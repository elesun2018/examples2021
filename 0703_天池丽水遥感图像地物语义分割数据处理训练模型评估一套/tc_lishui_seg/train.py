"""
Pytorch Unet模型
初赛：利用算法对遥感影像进行10大类地物要素分类，主要考察算法地物分类的准确性
ref:
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.6cc26423onxSOX&postId=169396
https://tianchi.aliyun.com/competition/entrance/531860/information
"""
import os
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import shutil
import torch
import torch.nn as nn
import torch.utils.data as D
torch.backends.cudnn.enabled = True
from torchvision import transforms as T
from PIL import Image

EPOCHES = 100
BATCH_SIZE = 16
IMAGE_SIZE = 256
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_dir = "models_train"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)

data_dir = "../../../datasets/seg/lishuitianchi/suichang_round1_train_210120" # ../../../results/lishui/sun/data_03021711/balance/*.tif results/lishui/sun/data_03021711  datasets/tc_lishui_clf/suichang_round1_train_210120/*.tif


class TianChiDataset(D.Dataset):
    def __init__(self, paths, test_mode=False):
        self.paths = paths
        self.test_mode = test_mode

        self.len = len(paths)
        self.transforms = T.Compose([
                                    T.ToPILImage(),
                                    T.Resize(IMAGE_SIZE),
                                    # T.RandomHorizontalFlip(p=0.5),
                                    # T.RandomVerticalFlip(p=0.5),
                                    T.ToTensor(),
                                    # T.Normalize([0.625, 0.448, 0.688],
                                    #             [0.131, 0.177, 0.101]),
        ])
    # get data operation
    def __getitem__(self, index):
        # img = cv2.imread(self.paths[index]) #mode:RGBNir
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(self.paths[index]) #mode:RGBNir
        img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        # print("img.shape", img.shape)  # (256, 256, 4)
        img = img[:, :, [2, 1, 3]]  # RGBNir - BGNir
        if not self.test_mode:
            mask = cv2.imread(self.paths[index].replace('.tif', '.png')) - 1
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
            return self.transforms(img), mask[:, :, 0].astype(np.int64)
        else:
            return self.transforms(img), ''

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

import glob
dataset = TianChiDataset(glob.glob(data_dir+"/*.tif"),False)

@torch.no_grad()
def validation(model, loader):
    val_iou = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output = output.argmax(1)
        iou = np_iou(output, target)
        val_iou.append(iou)
    return val_iou


def np_iou(pred, mask, c=10):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)

        uion = p.sum() + t.sum()
        overlap = (p * t).sum()

        # print(idx, uion, overlap)

        iou = 2 * overlap / (p.sum() + t.sum() + 0.001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

class_name = ['farm','land','forest','grass','road','urban_area',
                 'countryside','industrial_land','construction',
                 'water', 'bareland']
print("class_name : ",class_name)

header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
print("header\n",header)


valid_idx, train_idx = [], []

for i in range(len(dataset)):
    if i % 6 == 0:
        valid_idx.append(i)
    else :
        train_idx.append(i)

train_ds = D.Subset(dataset, train_idx)
valid_ds = D.Subset(dataset, valid_idx)

# define training and validation data loaders
loader = D.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

vloader = D.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

import segmentation_models_pytorch as smp
model = smp.DeepLabV3Plus( # elesun DeepLabV3Plus DeepLabV3 FPN Unet
        encoder_name="resnet101",        # resnet50 choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization  None
        in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=10,                      # model output channels (number of classes in your dataset)
)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4, weight_decay=1e-3)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)

best_iou = 0
for epoch in range(1, EPOCHES + 1):
    losses = []
    start_time = time.time()
    model.train()
    model.to(DEVICE)
    for image, target in tqdm(loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    viou = validation(model, vloader)
    print('\t'.join(np.stack(viou).mean(0).round(3).astype(str)))

    print(raw_line.format(epoch, np.array(losses).mean(), np.mean(viou),
                          (time.time() - start_time) / 60 ** 1))
    if best_iou < np.stack(viou).mean(0).mean():
        best_iou = np.stack(viou).mean(0).mean()
        model_save_path = os.path.join(model_dir, "model_valiou{0:.3f}_epoch{1:03}.pth".format(best_iou,epoch))
        torch.save(model.state_dict(), model_save_path)
        print("model saved in ", model_save_path) # 冒号后面指定输出宽度，：后的0表示用0占位。
