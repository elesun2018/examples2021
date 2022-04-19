"""
http请求 clent端
"""
import base64
import json
import numpy as np
import cv2
import requests
import os


def clent_post(url, image_path):
    send_dict = {}
    send_dict['test_ID'] = 31
    imageData = base64.b64encode(open(image_path, 'rb').read()).decode()
    send_dict['test_image'] = imageData
    send_json = json.dumps(send_dict)
    respon = requests.post(url=url, data=send_json, headers={'Content-Type': 'application/json'})
    res = json.loads(respon.text)
    return res

if __name__ == "__main__":
    url = "http://127.0.0.1:9730/FireReg" # 127.0.0.1 10.1.68.151
    image_path = "imgs/00000001.jpg" #
    res_path = "test_res_image.jpg"
    if os.path.isfile(res_path):
        os.remove(res_path)
    res_dict = clent_post(url, image_path)
    # print("res_dict", res_dict)
    # 获取响应字典字段
    print("test_ID", res_dict["test_ID"])
    print("test_status", res_dict["test_status"])
    print("test_result", res_dict["test_result"])
    print("test_time", res_dict["test_time"])
    # 获取响应字典图像字段
    test_res_image = res_dict['test_res_image']
    # print("type(test_res_image):", type(test_res_image))
    img_b64 = base64.b64decode(test_res_image)
    # print("type(img_b64):", type(img_b64))
    buff2np = np.frombuffer(img_b64, np.uint8)
    # print("type(buff2np):", type(buff2np))
    img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
    # print("type(img_dec):", type(img_dec))
    print("img_dec.shape", img_dec.shape)
    # cv2.namedWindow('test_res_image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('test_res_image', img_dec)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(res_path, img_dec)
    print("clent response end !")
