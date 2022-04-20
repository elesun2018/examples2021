# -*- coding: utf-8 -*-
"""
nam:
    
fun:
    
ref:
    
"""
import os
import time
import shutil
from tqdm import tqdm



def gen_trainvalset_afteryolo(in_label_dir, in_image_dir, in_set_dir,
                              out_label_dir, out_image_dir):
    if not os.path.exists(in_label_dir):
        raise ValueError(in_label_dir,"not exist!")
    if not os.path.exists(in_image_dir):
        raise ValueError(in_image_dir,"not exist!")
    if not os.path.exists(in_set_dir):
        raise ValueError(in_set_dir,"not exist!")
    set_list = ["train","val","trainval"]
    for set_name in set_list :
        print("processing set",set_name)
        out_image_set_dir = os.path.join(out_image_dir, set_name)
        if os.path.exists(out_image_set_dir):
            shutil.rmtree(out_image_set_dir)  # 递归删除文件夹
        os.makedirs(out_image_set_dir)
        out_label_set_dir = os.path.join(out_label_dir, set_name)
        if os.path.exists(out_label_set_dir):
            shutil.rmtree(out_label_set_dir)  # 递归删除文件夹
        os.makedirs(out_label_set_dir)
        save_set_path = os.path.join(out_image_dir, set_name + ".txt")
        if os.path.exists(save_set_path):
            os.remove(save_set_path)
        fw = open(save_set_path, 'w')
        in_set_path = os.path.join(in_set_dir, set_name + ".txt")
        if not os.path.exists(in_set_path):
            raise ValueError(in_set_path, "not exist!")
        with open(in_set_path, "r") as fp:
            lines = fp.read().splitlines()
        for img_name in tqdm(lines):
            source_image_path = os.path.join(in_image_dir, img_name + ".jpg")
            aim_image_path = os.path.join(out_image_set_dir, img_name + ".jpg")
            shutil.copy(source_image_path, aim_image_path)
            source_label_path = os.path.join(in_label_dir, img_name + ".txt")
            aim_label_path = os.path.join(out_label_set_dir, img_name + ".txt")
            shutil.copy(source_label_path, aim_label_path)
            img_path = os.path.join(os.path.abspath(out_image_dir), set_name, img_name + ".jpg\n")
            fw.write(img_path)
        fw.close()
    print(out_label_dir,"generated !")
    print(out_image_dir, "generated !")

if __name__=='__main__':
    time1 = time.time()
    in_label_dir = "VOCsportyololabels"
    in_image_dir = "../01processdata_genVOC/VOCsport/JPEGImages"
    in_set_dir = "../01processdata_genVOC/VOCsport/ImageSets/Main"
    out_label_dir = "VOCsport2yolo/labels"
    out_image_dir = "VOCsport2yolo/images"
    gen_trainvalset_afteryolo(in_label_dir = in_label_dir, in_image_dir = in_image_dir, in_set_dir = in_set_dir,
                              out_label_dir = out_label_dir, out_image_dir = out_image_dir)
    time2 = time.time()
    print(u'总共耗时：' + str(time2 - time1) + 's')

