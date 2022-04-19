# -*- coding: utf-8 -*-
"""

"""
import os
import random
random.seed(0)
import math
import time
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import xml.etree.ElementTree as ET

# 添加椒盐噪声
def sp_noise(noise_img, proportion):
    '''
    添加椒盐噪声
    proportion的值表示加入噪声的量，可根据需要自行调整
    return: img_noise
    '''
    height, width = noise_img.shape[0], noise_img.shape[1]#获取高度宽度像素值
    num = int(height * width * proportion) #一个准备加入多少噪声小点
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img
# 添加高斯噪声
def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out# 这里也会返回噪声，注意返回值
    
# rotate_img
def rotate_image(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    # convet angle into rad
    rangle = np.deg2rad(angle)  # angle in radians
    # calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # map
    return cv2.warpAffine(
        src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
        flags=cv2.INTER_LANCZOS4)

def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # get width and heigh of changed image
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # rot_mat: the final rot matrix
    # get the four center of edges in the initial martix，and convert the coord
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    # concat np.array
    concat = np.vstack((point1, point2, point3, point4))
    # change type
    concat = concat.astype(np.int32)
    # print(concat)
    rx, ry, rw, rh = cv2.boundingRect(concat)
    return rx, ry, rw, rh

    def process_img(self, imgs_path, xmls_path, img_save_path, xml_save_path, angle_list):
        filepath_lists = os.listdir(imgs_path)
        filepath_lists = [i.split(".xml")[0] for i in filepath_lists if (i.endswith(".xml"))]
        filepath_lists = [i for i in filepath_lists if os.path.isfile(os.path.join(imgs_path, i+".jpg"))]
        # assign the rot angles
        

def imgaug(in_dir, out_dir, imgs_file="JPEGImages", anns_file="Annotations"):
    if os.path.exists(in_dir):
        print("in_dir:",in_dir)
    else:
        print(in_dir,"not exist!")
        raise ValueError(in_dir,"not exist!")
    if os.path.exists(out_dir):
        print(out_dir,"exist and delete")
        shutil.rmtree(out_dir)  # 递归删除文件夹
    in_imgs_dir = os.path.join(in_dir, imgs_file)
    in_anns_dir = os.path.join(in_dir, anns_file)
    if not os.path.exists(in_imgs_dir):
        raise ValueError(in_imgs_dir,"not exist!")
    if not os.path.exists(in_anns_dir):
        raise ValueError(in_anns_dir,"not exist!")    
    out_imgs_dir = os.path.join(out_dir, imgs_file)
    out_anns_dir = os.path.join(out_dir, anns_file)
    print("makedirs:", out_imgs_dir)
    os.makedirs(out_imgs_dir)
    print("makedirs:", out_anns_dir)
    os.makedirs(out_anns_dir)
    
    time1 = time.time()
    filepath_lists = os.listdir(in_anns_dir)
    filepath_lists = [i.split(".xml")[0] for i in filepath_lists if (i.endswith(".xml"))]
    filepath_lists = [i for i in filepath_lists if os.path.isfile(os.path.join(in_imgs_dir, i+".jpg"))]
    print("filepath_lists length ",len(filepath_lists))
    
    for index,filename in enumerate(tqdm(filepath_lists)) :
        # print("index:",index," filename:",filename)
        img = cv2.imread(os.path.join(in_imgs_dir,filename+".jpg"))
        # 原图原标签拷贝
        cv2.imwrite(os.path.join(out_imgs_dir, filename+".jpg"),img)
        shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,filename+".xml"))
        img_cp1 = img.copy()
        img_cp2 = img.copy()
        img_cp3 = img.copy()
        img_cp4 = img.copy()
        # 高斯模糊化数据增强
        noisekernel_list = [3,7,11] # 3,5,7,9,11
        for size in noisekernel_list :
            img_blur = cv2.GaussianBlur(img, ksize=(size,size), sigmaX=0, sigmaY=0)
            savename = filename + '_blur' + str(size)
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_blur)
            shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,savename+".xml"))
        # 加入噪声数据增强
        proportion_list = [0.01,0.025,0.05] # 0.01,0.025,0.05
        for proportion in proportion_list :
            # img_noise = gaussian_noise(img, 0, 0.12) # 高斯噪声
            img_noise = sp_noise(img,proportion)# 椒盐噪声
            savename = filename + '_noise' + str(proportion*1000)
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_noise)
            shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,savename+".xml"))
        # HSV颜色空间转换数据增强  色调变化范围 [0， 179]
        # 通过cv2.cvtColor把图像从BGR转换到HSV
        img_hsv = cv2.cvtColor(img_cp1, cv2.COLOR_BGR2HSV)
        h_list = [-80,-40,-20,20,40,80] # -80,-40,-20,20,40,80
        for h in h_list :
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h) % 180
            img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 179)
            img_save = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            savename = filename + '_H' + str(h)
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_save)
            shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,savename+".xml"))
        # HSV颜色空间转换数据增强  饱和度变化范围 [0， 255]
        # 通过cv2.cvtColor把图像从BGR转换到HSV
        img_hsv = cv2.cvtColor(img_cp2, cv2.COLOR_BGR2HSV)
        s_list = [-40, -20, 40, 80] # -80,-40,-20,20,40,80
        s_list = [0.2, 0.5, 1.5, 2]
        for s in s_list :
            img_hsv[:, :, 1] = img_hsv[:, :, 1]*s
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
            img_save = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            savename = filename + '_S' + str(s)
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_save)
            shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,savename+".xml"))
        # 亮度调整数据增强
        bright_list = [2,2.5,3] # 2,2.5,3
        for gamma in bright_list :
            gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
            gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
            img_bright = cv2.LUT(img_cp3,gamma_table)
            savename = filename + '_bright' + str(int(gamma*10))
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_bright)
            shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,savename+".xml"))
        # 旋转角度数据增强
        angle_list = [90, 180, 270]
        for angle in angle_list:
            img_rotate = rotate_image(img_cp4, angle)
            savename = filename + '_angle' + str(angle)
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_rotate)
            xml_path = os.path.join(in_anns_dir,filename+".xml")
            tree = ET.parse(xml_path)
            file_name = tree.find('filename').text  # it is origin name
            path = tree.find('path').text  # it is origin path
            # change name and path
            tree.find('filename').text = savename  # change file name to rot degree name
            tree.find('path').text = savename  #  change file path to rot degree name
            # if angle in [90, 270], need to swap width and height
            if angle in [90, 270]:
                d = tree.find('size')
                width = int(d.find('width').text)
                height = int(d.find('height').text)
                # swap width and height
                d.find('width').text = str(height)
                d.find('height').text = str(width)
            root = tree.getroot()
            for ob in root.iter('object'):
                for name in ob.iter('name'):
                    name.text = name.text # 更新标签  str(angle)
                for box in ob.iter('bndbox'):
                    xmin = float(box.find('xmin').text)
                    ymin = float(box.find('ymin').text)
                    xmax = float(box.find('xmax').text)
                    ymax = float(box.find('ymax').text)
                    x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, angle)
                    # change the coord
                    box.find('xmin').text = str(x)
                    box.find('ymin').text = str(y)
                    box.find('xmax').text = str(x+w)
                    box.find('ymax').text = str(y+h)
                    # box.set('updated', 'yes')
            # write into new xml
            tree.write(os.path.join(out_anns_dir,savename+".xml"))

    time2 = time.time()
    print("time use {:.3f} s".format(time2 - time1))

if __name__=='__main__':
    imgaug(in_dir='smoke', out_dir="aug_smoke") # rotateaug_VOClabel



