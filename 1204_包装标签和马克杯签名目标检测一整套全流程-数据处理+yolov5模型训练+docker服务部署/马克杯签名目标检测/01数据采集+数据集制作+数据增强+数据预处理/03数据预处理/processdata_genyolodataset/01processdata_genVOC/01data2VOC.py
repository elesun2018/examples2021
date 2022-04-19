# -*- coding: utf-8 -*-
"""

"""
import os
import random
random.seed(0)
import time
import shutil
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

def data2VOC(in_dir="raw_dataset", out_dir = "VOC_dataset", imgs_file="JPEGImages", anns_file="Annotations", txts_file="ImageSets/Main", name_file="VOC.names"):
    if os.path.exists(in_dir):
        print("in_dir:",in_dir)
    else:
        print(in_dir,"not exist!")
        raise ValueError(in_dir,"not exist!")
    if os.path.exists(out_dir):
        print(out_dir,"exist and delete")
        shutil.rmtree(out_dir)  # 递归删除文件夹
    out_imgs_dir = os.path.join(out_dir, imgs_file)
    out_anns_dir = os.path.join(out_dir, anns_file)
    out_txts_dir = os.path.join(out_dir, txts_file)
    print("makedirs:", out_imgs_dir)
    os.makedirs(out_imgs_dir)
    print("makedirs:", out_anns_dir)
    os.makedirs(out_anns_dir)
    print("makedirs:", out_txts_dir)
    os.makedirs(out_txts_dir)
    trainvaltxt_path = os.path.join(out_txts_dir, "trainval.txt")
    fp_trainval = open(trainvaltxt_path, "w")
    traintxt_path = os.path.join(out_txts_dir, "train.txt")
    fp_train = open(traintxt_path, "w")
    valtxt_path = os.path.join(out_txts_dir, "val.txt")
    fp_val = open(valtxt_path, "w")
    name_path = os.path.join(out_dir, name_file)
    fp_name = open(name_path, "w")
    dict_class = {}
    class_lists = []
    time1 = time.time()
    filepath_lists = glob.glob(in_dir+"/*") # elesun LogoDet-3K\Leisure\bahama breeze\9.xml
    filepath_lists = [i.split(".xml")[0] for i in filepath_lists if (i.endswith(".xml"))]
    filepath_lists = [i for i in filepath_lists if os.path.exists(i+".xml") and os.path.exists(i+".jpg")]
    random.shuffle(filepath_lists)
    print("filepath_lists length ",len(filepath_lists))
    train_set = random.sample(filepath_lists, int(0.7*len(filepath_lists)))
    for index,filepath in enumerate(tqdm(filepath_lists)) :
        # print("index:",index," filepath:",filepath)
        filename = "%08d" % (index + 1)
        shutil.copy(filepath+".jpg", os.path.join(out_imgs_dir,filename+".jpg"))
        shutil.copy(filepath+".xml", os.path.join(out_anns_dir,filename+".xml"))
        # 写入ImageSets/Main/trainval.txt
        fp_trainval.write("%s\n"%(filename))
        # 写入ImageSets/Main/train.txt
        if filepath in train_set :
            fp_train.write("%s\n" % (filename))
        # 写入ImageSets/Main/val.txt
        else : # val_set
            fp_val.write("%s\n" % (filename))
        # 统计各类别
        tree = ET.parse(filepath+".xml")
        root = tree.getroot()
        for ob in root.iter('object'):
            for name in ob.iter('name'):
                label = name.text
                if label not in class_lists :
                    class_lists.append(label)
                    fp_name.write("%s\n" % (label))
                if label not in dict_class.keys() :
                    dict_class[label] = 0
                    # dict_class[label] = [filename]
                else :
                    dict_class[label] += 1
                    # dict_class[label].append(filename)
    fp_trainval.close()
    fp_train.close()
    fp_val.close()
    fp_name.close()
    time2 = time.time()
    print("time use {:.3f} s".format(time2 - time1))
    print("class_lists\n", class_lists)
    print("class_lists length ", len(class_lists))
    print("dict_class\n", dict_class)
    print("process rawdataset to VOCdataset end !")

    return dict_class

if __name__=='__main__':
    dict_class = data2VOC(in_dir="noiseaug_VOCmug", out_dir="VOCmug",name_file="VOCmug.names")



