
"""
Flask Web微框架 服务器端server
"""
import base64
import json
import time
from flask import Flask, request
import os
import numpy as np
import cv2
import logging
from logging.handlers import TimedRotatingFileHandler
from logging import handlers
with open("./config.json",'r') as load_f:
    load_dict = json.load(load_f)
    per_log_days = int(load_dict['per_log_days'])
    log_keep_days = int(load_dict['log_keep_days'])
def _logging(filename):
    level = logging.DEBUG
    datefmt = '%Y-%m-%d %H:%M:%S'
    format = '%(asctime)s [%(module)s] %(levelname)s [%(lineno)d] %(message)s'
    log = logging.getLogger(filename)
    format_str = logging.Formatter(format, datefmt)

    def namer(filename):
        return filename.split('default.')[1]

    th = handlers.TimedRotatingFileHandler(filename=filename, when='midnight', interval=per_log_days, backupCount=log_keep_days, encoding='utf-8')
    th.suffix = "%Y-%m-%d.log"
    th.setFormatter(format_str)
    th.setLevel(logging.INFO)
    log.addHandler(th)
    log.setLevel(level)
    return log  
os.makedirs('../logs', exist_ok=True)
log = _logging(filename='../logs/opencv')

app = Flask(__name__)
@app.route('/edge', methods=['POST'])
def edge():
    start_time = time.time()
    results_dict = {}
    # log.info("request.json %s",request.json)
    try:
        test_ID = request.json['test_ID']
        test_image = request.json['test_image']
        operation = request.json['operation']
        minval = int(request.json['parameter1'])
        maxval = int(request.json['parameter2'])
        log.info("rec all params")
        img_b64 = base64.b64decode(test_image)
        buff2np = np.frombuffer(img_b64, np.uint8)
        img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
        # print("img_dec.shape", img_dec.shape)  # 具有属性shape说明是各数组图片
        sp = img_dec.shape
        log.info("sp %s",sp)
    except: # error shape
        results_dict["test_status"] = -2 
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:    # 图像处理
        print(type(minval))
        out_img_cvnp = cv2.Canny(img_dec, minval, maxval)
    except: # error process
        results_dict["test_status"] = 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:  
        if out_img_cvnp is None :
            # print("no output")
            log.info("no output")
            raise Exception()
    except Exception : # error no output
        results_dict["test_status"] = -1
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:
        # 返回处理后的图像
        # print("type(out_img_cvnp):", type(out_img_cvnp))
        # print(out_img_cvnp.shape)
        log.info("out_img_cvnp.shape %s", out_img_cvnp.shape)
        out_img_cod = cv2.imencode('.jpg', out_img_cvnp)[1]
        # out_img_cod = cv2.imencode('.png', out_img_cvnp)
        # print("type(out_img_cod):", type(out_img_cod))
        out_img_b64 = base64.b64encode(out_img_cod)  # cv2.imencode('.jpg', out_img)[1].tobytes()
        # print("type(out_img_b64):",type(out_img_b64))
        results_dict['test_res_image'] = str(out_img_b64, encoding='utf-8')
        # print("type(results_dict[test_res_image]):", type(results_dict['test_res_image']))
        results_dict["test_status"] = 1
        results_dict["test_result"] = []
        results_dict["test_ID"] = test_ID
        end_time = time.time()
        test_time = round(end_time - start_time, 3)
        results_dict["test_time"] = test_time
    except: # error other
        results_dict["test_status"] = 0 # 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    # log.info("results_dict %s",results_dict)
    log.info("test_status %s",results_dict["test_status"])
    results_json = json.dumps(results_dict)
    # print("response time use {:.3f} s".format(test_time))
    return results_json

@app.route('/sharp', methods=['POST'])
def sharp():
    start_time = time.time()
    results_dict = {}
    # log.info("request.json %s",request.json)
    try:
        test_ID = request.json['test_ID']
        test_image = request.json['test_image']
        operation = request.json['operation']
        # min = request.json['parameter1']
        # max = request.json['parameter2']
        log.info("rec all params")
        img_b64 = base64.b64decode(test_image)
        buff2np = np.frombuffer(img_b64, np.uint8)
        img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
        # print("img_dec.shape", img_dec.shape)  # 具有属性shape说明是各数组图片
        sp = img_dec.shape
        log.info("sp %s",sp)
    except: # error shape
        results_dict["test_status"] = -2 
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:    # 图像处理
        matSharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(img_dec, cv2.CV_32F, matSharp)
        out_img_cvnp = cv2.convertScaleAbs(sharp)
    except: # error process
        results_dict["test_status"] = 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:  
        if out_img_cvnp is None :
            # print("no output")
            log.info("no output")
            raise Exception()
    except Exception : # error no output
        results_dict["test_status"] = -1
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:
        # 返回处理后的图像
        # print("type(out_img_cvnp):", type(out_img_cvnp))
        # print(out_img_cvnp.shape)
        log.info("out_img_cvnp.shape %s", out_img_cvnp.shape)
        out_img_cod = cv2.imencode('.jpg', out_img_cvnp)[1]
        # out_img_cod = cv2.imencode('.png', out_img_cvnp)
        # print("type(out_img_cod):", type(out_img_cod))
        out_img_b64 = base64.b64encode(out_img_cod)  # cv2.imencode('.jpg', out_img)[1].tobytes()
        # print("type(out_img_b64):",type(out_img_b64))
        results_dict['test_res_image'] = str(out_img_b64, encoding='utf-8')
        # print("type(results_dict[test_res_image]):", type(results_dict['test_res_image']))
        results_dict["test_status"] = 1
        results_dict["test_result"] = []
        results_dict["test_ID"] = test_ID
        end_time = time.time()
        test_time = round(end_time - start_time, 3)
        results_dict["test_time"] = test_time
    except: # error other
        results_dict["test_status"] = 0 # 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    # log.info("results_dict %s",results_dict)
    log.info("test_status %s",results_dict["test_status"])
    results_json = json.dumps(results_dict)
    # print("response time use {:.3f} s".format(test_time))
    return results_json

@app.route('/thresh', methods=['POST'])
def thresh():
    start_time = time.time()
    results_dict = {}
    # log.info("request.json %s",request.json)
    try:
        test_ID = request.json['test_ID']
        test_image = request.json['test_image']
        operation = request.json['operation']
        thresh = int(request.json['parameter1'])
        maxval = int(request.json['parameter2'])
        log.info("rec all params")
        img_b64 = base64.b64decode(test_image)
        buff2np = np.frombuffer(img_b64, np.uint8)
        img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
        # print("img_dec.shape", img_dec.shape)  # 具有属性shape说明是各数组图片
        sp = img_dec.shape
        log.info("sp %s",sp)
    except: # error shape
        results_dict["test_status"] = -2 
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:    # 图像处理
        # 图像处理
        #thresh = 67
        #maxval = 255
        # img = cv2.imread(img_path, 0)  # 直接读为灰度图像
        ret, out_img_cvnp = cv2.threshold(img_dec, thresh, maxval, cv2.THRESH_BINARY)
    except: # error process
        results_dict["test_status"] = 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:  
        if out_img_cvnp is None :
            # print("no output")
            log.info("no output")
            raise Exception()
    except Exception : # error no output
        results_dict["test_status"] = -1
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:
        # 返回处理后的图像
        # print("type(out_img_cvnp):", type(out_img_cvnp))
        # print(out_img_cvnp.shape)
        log.info("out_img_cvnp.shape %s", out_img_cvnp.shape)
        out_img_cod = cv2.imencode('.jpg', out_img_cvnp)[1]
        # out_img_cod = cv2.imencode('.png', out_img_cvnp)
        # print("type(out_img_cod):", type(out_img_cod))
        out_img_b64 = base64.b64encode(out_img_cod)  # cv2.imencode('.jpg', out_img)[1].tobytes()
        # print("type(out_img_b64):",type(out_img_b64))
        results_dict['test_res_image'] = str(out_img_b64, encoding='utf-8')
        # print("type(results_dict[test_res_image]):", type(results_dict['test_res_image']))
        results_dict["test_status"] = 1
        results_dict["test_result"] = []
        results_dict["test_ID"] = test_ID
        end_time = time.time()
        test_time = round(end_time - start_time, 3)
        results_dict["test_time"] = test_time
    except: # error other
        results_dict["test_status"] = 0 # 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    # log.info("results_dict %s",results_dict)
    log.info("test_status %s",results_dict["test_status"])
    results_json = json.dumps(results_dict)
    # print("response time use {:.3f} s".format(test_time))
    return results_json

@app.route('/filter', methods=['POST'])
def filter():
    start_time = time.time()
    results_dict = {}
    # log.info("request.json %s",request.json)
    try:
        test_ID = request.json['test_ID']
        test_image = request.json['test_image']
        operation = request.json['operation']
        kernel = int(request.json['parameter1'])
        # maxval = request.json['parameter2']
        log.info("rec all params")
        img_b64 = base64.b64decode(test_image)
        buff2np = np.frombuffer(img_b64, np.uint8)
        img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
        # print("img_dec.shape", img_dec.shape)  # 具有属性shape说明是各数组图片
        sp = img_dec.shape
        log.info("sp %s",sp)
    except: # error shape
        results_dict["test_status"] = -2 
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:    # 图像处理
        # 图像处理
        #kernel = 11  # 必须为奇数
        # out_img_cvnp = cv2.blur(img_dec, (kernel, kernel))  # 均值滤波
        # out_img_cvnp = cv2.GaussianBlur(img_dec, (kernel, kernel), 0)  # 高斯滤波
        out_img_cvnp = cv2.medianBlur(img_dec, kernel)  # 中值滤波
        # 返回处理后的图像
    except: # error process
        results_dict["test_status"] = 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:  
        if out_img_cvnp is None :
            # print("no output")
            log.info("no output")
            raise Exception()
    except Exception : # error no output
        results_dict["test_status"] = -1
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:
        # 返回处理后的图像
        # print("type(out_img_cvnp):", type(out_img_cvnp))
        # print(out_img_cvnp.shape)
        log.info("out_img_cvnp.shape %s", out_img_cvnp.shape)
        out_img_cod = cv2.imencode('.jpg', out_img_cvnp)[1]
        # out_img_cod = cv2.imencode('.png', out_img_cvnp)
        # print("type(out_img_cod):", type(out_img_cod))
        out_img_b64 = base64.b64encode(out_img_cod)  # cv2.imencode('.jpg', out_img)[1].tobytes()
        # print("type(out_img_b64):",type(out_img_b64))
        results_dict['test_res_image'] = str(out_img_b64, encoding='utf-8')
        # print("type(results_dict[test_res_image]):", type(results_dict['test_res_image']))
        results_dict["test_status"] = 1
        results_dict["test_result"] = []
        results_dict["test_ID"] = test_ID
        end_time = time.time()
        test_time = round(end_time - start_time, 3)
        results_dict["test_time"] = test_time
    except: # error other
        results_dict["test_status"] = 0 # 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    # log.info("results_dict %s",results_dict)
    log.info("test_status %s",results_dict["test_status"])
    results_json = json.dumps(results_dict)
    # print("response time use {:.3f} s".format(test_time))
    return results_json

@app.route('/edit', methods=['POST'])
def edit():
    start_time = time.time()
    results_dict = {}
    # log.info("request.json %s",request.json)
    try:
        test_ID = request.json['test_ID']
        test_image = request.json['test_image']
        operation = request.json['operation']
        parameter1 = int(request.json['parameter1'])
        parameter2 = int(request.json['parameter2'])
        log.info("rec all params")
        img_b64 = base64.b64decode(test_image)
        buff2np = np.frombuffer(img_b64, np.uint8)
        img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
        # print("img_dec.shape", img_dec.shape)  # 具有属性shape说明是各数组图片
        sp = img_dec.shape
        log.info("sp %s",sp)
    except: # error shape
        results_dict["test_status"] = -2 
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:    # 图像处理
        h = img_dec.shape[0]
        w = img_dec.shape[1]
        if operation == "gray":
            out_img_cvnp = cv2.cvtColor(img_dec, cv2.COLOR_BGR2GRAY)  #
        elif operation == "scale":
            out_img_cvnp = cv2.resize(img_dec, (0,0), fx=parameter1,fy=parameter2)  # interpolation=cv2.INTER_LINEAR
        elif operation == "rotate":
            angle = parameter1
            matRotate = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1)  # 旋转变化矩阵
            out_img_cvnp = cv2.warpAffine(img_dec, matRotate, (w, h))  # 旋转
        elif operation == "crop":
            crop_y0 = int(0.5*h)
            crop_x0 = int(0.5*w)
            crop_x1 = parameter1
            crop_y1 = parameter2
            out_img_cvnp = img_dec[int(crop_y0):int(crop_y1),int(crop_x0):int(crop_x1)] #区域裁剪
        elif operation == "shift":
            shift_x = parameter1
            shift_y = parameter2
            matShift = np.float32([[1, 0, shift_x], [0, 1, shift]])  # 偏移变化矩阵
            out_img_cvnp = cv2.warpAffine(img_dec, matShift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # 偏移
        else :
            print("no such operation",operation)
        # 返回处理后的图像
    except: # error process
        results_dict["test_status"] = 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:  
        if out_img_cvnp is None :
            # print("no output")
            log.info("no output")
            raise Exception()
    except Exception : # error no output
        results_dict["test_status"] = -1
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    try:
        # 返回处理后的图像
        # print("type(out_img_cvnp):", type(out_img_cvnp))
        # print(out_img_cvnp.shape)
        log.info("out_img_cvnp.shape %s", out_img_cvnp.shape)
        out_img_cod = cv2.imencode('.jpg', out_img_cvnp)[1]
        # out_img_cod = cv2.imencode('.png', out_img_cvnp)
        # print("type(out_img_cod):", type(out_img_cod))
        out_img_b64 = base64.b64encode(out_img_cod)  # cv2.imencode('.jpg', out_img)[1].tobytes()
        # print("type(out_img_b64):",type(out_img_b64))
        results_dict['test_res_image'] = str(out_img_b64, encoding='utf-8')
        # print("type(results_dict[test_res_image]):", type(results_dict['test_res_image']))
        results_dict["test_status"] = 1
        results_dict["test_result"] = []
        results_dict["test_ID"] = test_ID
        end_time = time.time()
        test_time = round(end_time - start_time, 3)
        results_dict["test_time"] = test_time
    except: # error other
        results_dict["test_status"] = 0 # 0
        results_dict["test_result"] = []
        results_dict["test_time"] = ""
        results_dict["test_res_image"] = ""
        # log.info("results_dict %s",results_dict)
        results_json = json.dumps(results_dict)
        return results_json
    # log.info("results_dict %s",results_dict)
    log.info("test_status %s",results_dict["test_status"])
    results_json = json.dumps(results_dict)
    # print("response time use {:.3f} s".format(test_time))
    return results_json

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8060)
