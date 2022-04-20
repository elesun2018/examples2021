# -*- coding: utf-8 -*-
"""
模型预测服务
"""
import os
import numpy as np
import cv2
from PIL import Image

def fun(img, model):
	if not os.path.exists(model):
		raise ValueError(model, "not exist !")
	if img is None:
		raise ValueError("infer img data not exist !")
	res_img_path = "img_rec.jpg"
	res_img = Image.open(res_img_path)
	res_img = np.asarray(res_img)
	# print("res_img.shape", res_img.shape)
	res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
	# print("res_img.shape", res_img.shape)
	res_dict = {
		0 : ['其他', '黑色', 4616404],
		1 : ['烤烟', '蓝色', 8806600],
		2 : ['玉米', '绿色', 844266],
		3 : ['水稻', '红色', 630970]
	}
	return res_dict,res_img

if __name__ == "__main__":
	img_path = "img_rec.jpg"  # 注意修改路径
	in_img = Image.open(img_path)
	in_img = np.asarray(in_img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
	# print("img.shape", img.shape)
	in_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
	# print("img.shape", img.shape)
	res_dict,out_img = fun(in_img, model="model.pth")
	print("res_dict",res_dict)
	if out_img is not None:
		print("out_img.shape", out_img.shape)





