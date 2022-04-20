"""
Pytorch Unet模型
初赛：利用算法对遥感影像进行10大类地物要素分类，主要考察算法地物分类的准确性
ref:
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.6cc26423onxSOX&postId=169396
https://tianchi.aliyun.com/competition/entrance/531860/information
"""
import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
import shutil

import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
torch.backends.cudnn.enabled = True

import torchvision
from torchvision import transforms as T

# EPOCHES = 20
BATCH_SIZE = 16
IMAGE_SIZE = 256
# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])


class TianChiDataset(D.Dataset):
    def __init__(self, paths, transform, test_mode=False):
        self.paths = paths
        self.transform = transform
        self.test_mode = test_mode

        self.len = len(paths)
        self.as_tensor = T.Compose([
                                    T.ToPILImage(),
                                    T.Resize(IMAGE_SIZE),
                                    # T.ToTensor(),
                                    # T.Normalize([0.625, 0.448, 0.688],
                                    #             [0.131, 0.177, 0.101]),
        ])

    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.test_mode:
            mask = cv2.imread(self.paths[index].replace('.tif', '.png')) - 1
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][:, :, 0].astype(np.int64)
        else:
            return self.as_tensor(img), ''

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

import glob
dataset = TianChiDataset(
    glob.glob('../../../datasets/seg/lishuitianchi/suichang_round1_train_210120/*.tif'),
    trfm, False
)

out_dir = "../../../results/lishui/visual"
save_img_dir = os.path.join(out_dir, "compare_baseline")
if os.path.exists(save_img_dir):
    shutil.rmtree(save_img_dir)
os.makedirs(save_img_dir)
for i in tqdm(range(len(dataset))): # elesun 16017
    image, mask = dataset[i] # 150
    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.imshow(mask, cmap='gray')
    plt.subplot(122)
    plt.imshow(image)
    plt.savefig(os.path.join(save_img_dir,"compare_%05d_.png"%(i)))