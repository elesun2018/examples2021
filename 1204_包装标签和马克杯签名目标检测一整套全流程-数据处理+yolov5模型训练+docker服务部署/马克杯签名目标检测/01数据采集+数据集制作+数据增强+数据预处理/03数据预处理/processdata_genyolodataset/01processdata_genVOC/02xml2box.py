#!/usr/bin/env python3
# coding=utf-8
import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def detection_result_visual(img_path, label_path, out_data_visual):
    """
    可视化数据集
    :param img_path: 原始图片文件
    :param label_path: xml标签
    :param out_data_visual: 将xml的标签可视化到图片文件上
    :return:
    """
    voc_xml = label_path
    voc_img = img_path
    xml_path_ = os.listdir(voc_xml)
    if os.path.exists(out_data_visual):
        shutil.rmtree(out_data_visual)
    os.mkdir(out_data_visual)

    for xml_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
        xml_pth = os.path.join(voc_xml, xml_name)
        img_name = str(xml_name.replace('xml', 'jpg'))
        img_pth = os.path.join(voc_img, img_name)
        print("processing ",img_pth)
        img = Image.open(img_pth)
        draw = ImageDraw.Draw(img)
        tree = ET.parse(xml_pth)
        box = {}
        root = tree.getroot()
        for ob in root.iter('object'):
            for name in ob.iter('name'):
                label = name.text # label_dict[name.text]
            # for score in ob.iter('score'):
            #     score = score.text
            for bndbox in ob.iter('bndbox'):
                for xmin in bndbox.iter('xmin'):
                    box['xmin'] = float(xmin.text)
                for ymin in bndbox.iter('ymin'):
                    box['ymin'] = float(ymin.text)
                for xmax in bndbox.iter('xmax'):
                    box['xmax'] = float(xmax.text)
                for ymax in bndbox.iter('ymax'):
                    box['ymax'] = float(ymax.text)
                draw.line((box['xmin'], box['ymin'], box['xmax'], box['ymin']), fill=(255, 0, 0), width=4)
                draw.line((box['xmax'], box['ymin'], box['xmax'], box['ymax']), fill=(255, 0, 0), width=4)
                draw.line((box['xmax'], box['ymax'], box['xmin'], box['ymax']), fill=(255, 0, 0), width=4)
                draw.line((box['xmin'], box['ymax'], box['xmin'], box['ymin']), fill=(255, 0, 0), width=4)
                draw.text((box['xmin'], box['ymin'] - 22), label, fill="#00ffff", )
                # draw.text((box['xmin'], box['ymin'] - 12), score, fill="#00ffff", )
        img.save(os.path.join(out_data_visual, img_name))


if __name__ == '__main__':
    img_path = "VOCmug/JPEGImages"
    label_path = "VOCmug/Annotations"
    out_dir = "vis_output_VOCmug"
    # 可视化数据集
    detection_result_visual(img_path, label_path, out_dir)
