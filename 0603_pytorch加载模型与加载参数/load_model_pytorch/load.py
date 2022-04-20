import torch
from models.yolo import Model
cfg_dict = {
        "nc": 80,
        "depth_multiple": 0.33,  # model depth multiple
        "width_multiple": 0.50 , # layer channel multiple
        "anchors":[
          [10,13, 16,30, 33,23],  # P3/8
          [30,61, 62,45, 59,119],  # P4/16
          [116,90, 156,198, 373,326]  # P5/32
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
# 根据参数创建网络
network = Model(cfg_dict, ch=3, nc=20) # "models/yolov5s.yaml" cfg_dict
# 加载模型文件（已训练完成）
model_path = "weights/voc-yolov5s-best.pt"
ckpt = torch.load(model_path)
# 模型文件保存了键值对
print("ckpt.keys : ",ckpt.keys())
# 真正的参数模型在keys model
model = ckpt['model']
# 获取参数模型中的参数键值对数据
state_dict = model.state_dict()
print("state_dict.keys : ", state_dict.keys())
# 网络加载从参数数据
network.load_state_dict(state_dict)
print("load model end!")