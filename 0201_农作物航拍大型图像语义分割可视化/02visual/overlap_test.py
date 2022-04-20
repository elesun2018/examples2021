# -*- coding: utf-8 -*-
'''
两张图片叠加显示
https://blog.csdn.net/l641208111/article/details/106202138
'''
import cv2

mat_img = cv2.imread( "000999.jpg")
label_img = cv2.imread( "000999.png")
dist = cv2.addWeighted(mat_img, 0.6, label_img, 0.4, 0)
cv2.imwrite("000999_overlap.jpg", dist)

mat_img = cv2.imread( "train_image_10.png")
label_img = cv2.imread( "train_image_10_label.png")
dist = cv2.addWeighted(mat_img, 0.5, label_img, 0.5, 0)
cv2.imwrite("train_image_10_overlap.jpg", dist)

