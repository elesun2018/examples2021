
"""
Flask Web微框架 服务器端server
"""
import base64
import json
import time

import cv2
import numpy as np
from flask import Flask, request
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # sun
from models.yolo import Model
import numpy as np
import cv2
import torch
from numpy import random
import shutil
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box

app = Flask(__name__)

name_classes = ['0']

class YOLO(object):
    def __init__(self, model_path):
        self.model_path = model_path
        #################################GPU config#############################################
        print("#" * 30, "GPU config", "#" * 30)
        print("torch version :", torch.__version__)
        print("cuda version :", torch.version.cuda)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # elesun
        print("GPU available :", torch.cuda.is_available())
        print("GPU count :", torch.cuda.device_count())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda cuda:0 cuda:1 cpu
        print("GPU current :", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
        print("device: ", device)
        self.device = device
        #################################load model#############################################
        print("#" * 30, "model load", "#" * 15)
        start_time = time.time()
        cfg_dict = {
            "nc": 1,
            "depth_multiple": 0.33,  # model depth multiple
            "width_multiple": 0.50,  # layer channel multiple
            "anchors": [
                [10, 13, 16, 30, 33, 23],  # P3/8
                [30, 61, 62, 45, 59, 119],  # P4/16
                [116, 90, 156, 198, 373, 326]  # P5/32
            ],
            "backbone":  # [from, number, module, args]
                [[-1, 1, "Focus", [64, 3]],  # 0-P1/2
                 [-1, 1, "Conv", [128, 3, 2]],  # 1-P2/4
                 [-1, 3, "C3", [128]],
                 [-1, 1, "Conv", [256, 3, 2]],  # 3-P3/8
                 [-1, 9, "C3", [256]],
                 [-1, 1, "Conv", [512, 3, 2]],  # 5-P4/16
                 [-1, 9, "C3", [512]],
                 [-1, 1, "Conv", [1024, 3, 2]],  # 7-P5/32
                 [-1, 1, "SPP", [1024, [5, 9, 13]]],
                 [-1, 3, "C3", [1024, False]],  # 9
                 ],
            "head":
                [[-1, 1, "Conv", [512, 1, 1]],
                 [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
                 [[-1, 6], 1, "Concat", [1]],  # cat backbone P4
                 [-1, 3, "C3", [512, False]],  # 13

                 [-1, 1, "Conv", [256, 1, 1]],
                 [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
                 [[-1, 4], 1, "Concat", [1]],  # cat backbone P3
                 [-1, 3, "C3", [256, False]],  # 17 (P3/8-small)

                 [-1, 1, "Conv", [256, 3, 2]],
                 [[-1, 14], 1, "Concat", [1]],  # cat head P4
                 [-1, 3, "C3", [512, False]],  # 20 (P4/16-medium)

                 [-1, 1, "Conv", [512, 3, 2]],
                 [[-1, 10], 1, "Concat", [1]],  # cat head P5
                 [-1, 3, "C3", [1024, False]],  # 23 (P5/32-large)

                 [[17, 20, 23], 1, "Detect", ["nc", "anchors"]],  # Detect(P3, P4, P5)
                 ]
        }
        self.model = Model(cfg_dict, ch=3, nc=1)
        ckpt = torch.load(self.model_path)
        state_dict = ckpt['model'].state_dict()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        end_time = time.time()
        print("load model time use {:.3f} S".format(end_time - start_time))
    def infer_imgdata(self, image_name, img_dec):
        #################################model infer#############################################
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        print("#" * 30, "model infer", "#" * 15)
        res_dict_list = []
        start_time = time.time()
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in name_classes]
        conf_thres = 0.25 # 0.25
        iou_thres = 0.45 # 0.45
        img_size = 640
        stride = 32
        with torch.no_grad():
            img0 = img_dec
            # Padded resize
            img = letterbox(img0, img_size, stride=stride)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = img.unsqueeze(0)
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            for i, det in enumerate(pred):  # detections per image
                save_img_path = os.path.join(result_dir, image_name)
                boxes_str = ''
                boxes_str += '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    # print("box len",len(det))
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        boxes_str += f"{n} {name_classes[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # print("boxes_str :", boxes_str)
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line = (cls, *xywh, conf)
                        # print("line",line) # (tensor(14., device='cuda:1'), 0.49531251192092896, 0.637499988079071, 0.776562511920929, 0.7250000238418579, tensor(0.54114, device='cuda:1'))
                        # Add bbox to image
                        tag = f'{name_classes[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=tag, color=colors[int(cls)], line_thickness=3)
                        # Add bbox to xml
                        label = name_classes[int(cls)]
                        score = float('%.3f' % conf)
                        xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        res_dict = {}
                        res_dict["image_name"] = image_name
                        res_dict["ID"] = label
                        res_dict["probability"] = score
                        res_dict["location"] = {"width": xmax-xmin, "top": ymin, "height": ymax-ymin, "left": xmin}
                        res_dict_list.append(res_dict)
                cv2.imwrite(save_img_path, img0)
        end_time = time.time()
        print("infer time use {:.3f} S".format(end_time - start_time))
        # return res_dict_list
        if len(res_dict) > 0 :
            return img0
        else :
            return None

@app.route('/FireReg', methods=['POST'])
def FireReg():
    start_time = time.time()
    results_dict = {}
    try:
        test_ID = request.json['test_ID']
        print("test_ID",test_ID)
        test_image = request.json['test_image']
        img_b64 = base64.b64decode(test_image)
        buff2np = np.frombuffer(img_b64, np.uint8)
        img_dec = cv2.imdecode(buff2np, cv2.IMREAD_COLOR)
        print("img_dec.shape", img_dec.shape)  # 具有属性shape说明是各数组图片
        results_dict["test_ID"] = test_ID
        # 模型推理预测
        image_name = "temp_imgname.jpg"
        out_img_cvnp = yolo.infer_imgdata(image_name, img_dec)
        results_dict["test_status"] = 1  # 成功
        if out_img_cvnp is None :
            results_dict["test_result"] = 0  # 识别结果，0：无火灾隐患，1：有火灾隐患
        else :
            results_dict["test_result"] = 1  # 识别结果，0：无火灾隐患，1：有火灾隐患
        # 返回处理后的图像
        # print("type(out_img_cvnp):", type(out_img_cvnp))
        out_img_cod = cv2.imencode('.jpg', out_img_cvnp)[1]
        # out_img_cod = cv2.imencode('.png', out_img_cvnp)
        # print("type(out_img_cod):", type(out_img_cod))
        out_img_b64 = base64.b64encode(out_img_cod)  # cv2.imencode('.jpg', out_img)[1].tobytes()
        # print("type(out_img_b64):",type(out_img_b64))
        results_dict['test_res_image'] = str(out_img_b64, encoding='utf-8')
        # print("type(results_dict[test_res_image]):", type(results_dict['test_res_image']))

        end_time = time.time()
        test_time = round(end_time - start_time, 3)
        results_dict["test_time"] = test_time
        results_json = json.dumps(results_dict)
    except:
        results_dict["test_status"] = 0  # 失败
        results_dict["test_result"] = -1  # 失败：-1
        results_dict["test_ID"] = test_ID
        results_json = json.dumps(results_dict)
        return results_json
    print("response time use {:.3f} s".format(test_time))
    return results_json

if __name__ == '__main__':
    model_path = "model_loc/fire-yolov5s-best.pt"
    yolo = YOLO(model_path)
    app.run(host='0.0.0.0', port=9730)
