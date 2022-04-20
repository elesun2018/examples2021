"""
Pytorch Unet模型
初赛：利用算法对遥感影像进行10大类地物要素分类，主要考察算法地物分类的准确性
ref:
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.6cc26423onxSOX&postId=169396
https://tianchi.aliyun.com/competition/entrance/531860/information
"""
import sys, os, random, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import torch
torch.backends.cudnn.enabled = True
import glob
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3' # 可见哪些GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = 256
curr_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(curr_time,'%m%d%H%M')
model_path = "models_train/model_valiou0.424_epoch004.pth"
data_dir = 'dataset/val'
result_submit_img_dir = "results/submit/result_deplab3p_res101_"+time_str
result_visual_img_dir = "results/visual/compare_deplab3p_res101_"+time_str

if os.path.exists(result_submit_img_dir):
    shutil.rmtree(result_submit_img_dir)
os.makedirs(result_submit_img_dir)
if os.path.exists(result_visual_img_dir):
    shutil.rmtree(result_visual_img_dir)
os.makedirs(result_visual_img_dir)


trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

model = smp.DeepLabV3Plus(
            encoder_name="resnet101",        # "resnet50" choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # "imagenet" use `imagenet` pretreined weights for encoder initialization
            in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=10,                      # model output channels (number of classes in your dataset)
    )
model.load_state_dict(torch.load(model_path,map_location=DEVICE))
model.to(DEVICE)
model.eval()

start_time = time.time()
for idx, img_name in enumerate(tqdm(glob.glob(data_dir+"/*.tif")[:])):
    image = cv2.imread(img_name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]
        score1 = model(image).cpu().numpy()
        score2 = model(torch.flip(image, [0, 3]))
        #         score2 = score2.cpu().numpy()
        score2 = torch.flip(score2, [3, 0]).cpu().numpy()
        score3 = model(torch.flip(image, [0, 2]))
        #         score3 = score3.cpu().numpy()
        score3 = torch.flip(score3, [2, 0]).cpu().numpy()
        score = (score1 + score2 + score3) / 3.0
        score_sigmoid = score[0].argmax(0) + 1
        # 绘图
        # score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
        plt.figure(figsize=(16,8))
        plt.subplot(151)
        plt.imshow((score1[0].argmax(0) + 1)*30, cmap='gray')
        plt.subplot(152)
        plt.imshow((score2[0].argmax(0) + 1)*30, cmap='gray')
        plt.subplot(153)
        plt.imshow((score3[0].argmax(0) + 1)*30, cmap='gray')
        plt.subplot(154)
        plt.imshow((score[0].argmax(0) + 1)*30, cmap='gray')
        plt.subplot(155)
        image = cv2.imread(img_name)
        plt.imshow(image)
        vis_img_path = os.path.join(result_visual_img_dir,img_name.split('/')[-1].replace('.tif', '.png'))
        plt.savefig(vis_img_path)  # 保存图片
        # print(score_sigmoid.min(), score_sigmoid.max()) # 1 9
        print("values in predict : ", np.unique(score_sigmoid))
        cv2.imwrite(os.path.join(result_submit_img_dir,img_name.split('/')[-1].replace('.tif', '.png')), score_sigmoid)
end_time = time.time()
print("time use : ", round((end_time - start_time), 3))
# score.shape, score3.shape
# !zip -r results.zip results