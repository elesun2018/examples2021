# -*-coding:utf-8-*-

import argparse
import os

from Format import VOC, COCO, UDACITY, KITTI, YOLO

parser = argparse.ArgumentParser(description='label Converting example.')
parser.add_argument('--datasets', default="VOC", type=str, help='type of datasets')
parser.add_argument('--in_img_path', default="../01processdata_genVOC/VOCmug/JPEGImages", type=str, help='directory of image folder')
parser.add_argument('--img_type', default=".jpg", type=str, help='type of image')
parser.add_argument('--in_label', default="../01processdata_genVOC/VOCmug/Annotations", type=str,
                    help='directory of label folder or label file path')
parser.add_argument('--in_cls_list_file', type=str, help='directory of *.names file', default="VOCmug.names")
parser.add_argument('--output_label_path', default="VOCmugyololabels", type=str, help='directory of label folder')
parser.add_argument('--output_manipast_path', type=str, help='path of manifast file', default="manifast_VOCmug.txt")
args = parser.parse_args()


def main(config):
    if config["datasets"] == "VOC":
        voc = VOC()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = voc.parse(config["label"])

        if flag == True:

            flag, data = yolo.generate(data)
            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manipast_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))


        else:
            print("VOC Parsing Result : {}, msg : {}".format(flag, data))


    elif config["datasets"] == "COCO":
        coco = COCO()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = coco.parse(config["label"])

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manipast_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("COCO Parsing Result : {}, msg : {}".format(flag, data))

    elif config["datasets"] == "UDACITY":
        udacity = UDACITY()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = udacity.parse(config["label"])

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manipast_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("UDACITY Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("COCO Parsing Result : {}, msg : {}".format(flag, data))

    elif config["datasets"] == "KITTI":
        kitti = KITTI()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = kitti.parse(config["label"], config["img_path"], img_type=config["img_type"])

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manipast_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("KITTI Parsing Result : {}, msg : {}".format(flag, data))

    else:
        print("Unkwon Datasets")


if __name__ == '__main__':
    config = {
        "datasets": args.datasets,
        "img_path": args.in_img_path,
        "img_type": args.img_type,
        "label": args.in_label,
        "cls_list": args.in_cls_list_file,
        "output_path": args.output_label_path,
        "manipast_path": args.output_manipast_path,
    }

    main(config)
