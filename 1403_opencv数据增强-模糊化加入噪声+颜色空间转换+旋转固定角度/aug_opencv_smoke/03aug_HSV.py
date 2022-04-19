# -*- coding: utf-8 -*-
"""
图片处理，HSV、色调、亮度调节
https://blog.csdn.net/qq_21237549/article/details/121277920
"""
import cv2

img=cv2.imread('smoke/JPEGImages/759.jpg', cv2.IMREAD_COLOR)    # 1 759

# 通过cv2.cvtColor把图像从BGR转换到HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# H空间中，绿色比黄色的值高一点，所以给每个像素+15，黄色的树叶就会变绿
turn_green_hsv = img_hsv.copy()
turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0]+15) % 180
turn_green_img = cv2.cvtColor(turn_green_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('turn_green_img',turn_green_img)
cv2.imwrite('turn_green_img.jpg', turn_green_img)
cv2.waitKey(0)
# 减小饱和度会让图像损失鲜艳，变得更灰
colorless_hsv = img_hsv.copy()
colorless_hsv[:, :, 1] = 0.5 * colorless_hsv[:, :, 1]
colorless_img = cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('colorless_img',colorless_img)
cv2.waitKey(0)
cv2.imwrite('colorless_img.jpg', colorless_img)

# 调整明度  
darker_hsv = img_hsv.copy()
darker_hsv[:, :, 2] =0.5 * darker_hsv[:, :, 2] # 255全为最大，相当于只在意颜色种类，与光线无关了
darker_img = cv2.cvtColor(darker_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('darker_img',darker_img)
cv2.waitKey(0)
cv2.imwrite('darker_img.jpg', darker_img)