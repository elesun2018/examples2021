# -*- coding: utf-8 -*-
"""
Flask Web微框架 服务器端server
"""
import numpy as np
import time
import inference
import json
import base64
from flask import Flask,request
import cv2

app = Flask(__name__)

@app.route('/', methods=['POST'])
def server_response():
	start_time = time.time()
	results_dict = {}
	try:
		test_ID = request.json['test_ID']
		img_json = request.json['test_image']
	except:
		results_dict["test_status"] = -1
		results_dict["test_result"] = 'error'
		results_dict["test_time"] = 0
		results_json = json.dumps(results_dict)
		return results_json
	print("type(img_json):", type(img_json))
	img_b64 = base64.b64decode(img_json)
	print("type(img_b64):", type(img_b64))
	buff2np = np.frombuffer(img_b64, np.uint8)
	print("type(buff2np):", type(buff2np))
	img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
	print("type(img_dec):", type(img_dec))
	print("img_dec.shape",img_dec.shape)
	res_dict,out_img_cvnp = inference.fun(img_dec, model="model.pth")
	results_dict["test_result"] = res_dict
	# print("type(out_img_cvnp):", type(out_img_cvnp))
	out_img_cod = cv2.imencode('.jpg', out_img_cvnp)[1]
	# out_img_cod = cv2.imencode('.png', out_img_cvnp)
	# print("type(out_img_cod):", type(out_img_cod))
	out_img_b64 = base64.b64encode(out_img_cod) # cv2.imencode('.jpg', out_img)[1].tobytes()
	# print("type(out_img_b64):",type(out_img_b64))
	results_dict['test_res_image'] = str(out_img_b64, encoding='utf-8')
	# print("type(results_dict[test_res_image]):", type(results_dict['test_res_image']))
	end_time = time.time()
	test_time = round(end_time - start_time,4)
	results_dict["test_ID"] = test_ID
	results_dict["test_status"] = 0
	results_dict["test_time"] = test_time
	results_json = json.dumps(results_dict) #将结果填入json文件中并返回结果
	return results_json

if __name__ == '__main__':
	app.run(host='127.0.0.1',port=6651)
