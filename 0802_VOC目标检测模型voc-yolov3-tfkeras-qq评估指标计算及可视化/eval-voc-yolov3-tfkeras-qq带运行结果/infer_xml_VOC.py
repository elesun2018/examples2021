import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # sun
import shutil
from lxml import etree
import colorsys
import numpy as np
from keras import backend as K
from keras.layers import Input
from PIL import Image
from network.model import yolo_eval, yolo_body, tiny_yolo_body
from network.utils import letterbox_image
import os

class YOLO(object):
    def __init__(self, model_path, name_classes, anchors_np, score, iou, model_image_size):
        self.model_path = model_path
        self.score = score
        self.iou = iou
        self.model_image_size = model_image_size
        self.class_names = name_classes
        self.anchors = anchors_np
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        # start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print("img.shape:",image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        boxes_str_list = []  # sun
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            ymin, xmin, ymax, xmax = box # sunsun
            ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
            xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
            ymax = min(image.size[1], np.floor(ymax + 0.5).astype('int32'))
            xmax = min(image.size[0], np.floor(xmax + 0.5).astype('int32'))
            box_str = predicted_class + "," + str(score) \
                      + "," + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "\n"
            # print("box_str", box_str)  # sun
            boxes_str_list.append(box_str)
        # end = timer()
        # print("use time :",end - start)
        return "".join(boxes_str_list)# sun
    def close_session(self):
        self.sess.close()

class GEN_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
        child3 = etree.SubElement(self.root, "source")
        child4 = etree.SubElement(child3, "database")
        child4.text = "The PASCAL VOC2007 Database"
        child5 = etree.SubElement(child3, "annotation")
        child5.text = "PASCAL VOC2007"
        child6 = etree.SubElement(child3, "image")
        child6.text = "flickr"
        child7 = etree.SubElement(child3, "flickrid")
        child7.text = "69965"
        child8 = etree.SubElement(self.root, "owner")
        child9 = etree.SubElement(child8, "flickrid")
        child9.text = "sun"
        child10= etree.SubElement(child8, "name")
        child10.text = "sfz"
    def set_size(self,witdh,height,channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)
    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')
    def add_pic_attr(self,label,score,xmin,ymin,xmax,ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        pose = etree.SubElement(object, "score")
        pose.text = score
        truncated = etree.SubElement(object, "truncated")
        truncated.text = str(0)
        difficult = etree.SubElement(object, "difficult")
        difficult.text = str(0)
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)
    def genvoc(filename,class_,width,height,depth,xmin,ymin,xmax,ymax,savedir):
        anno = GEN_Annotations(filename)
        anno.set_size(width,height,depth)
        anno.add_pic_attr("pos", xmin, ymin, xmax, ymax)
        anno.savefile(savedir)

def model_infer(model_path, test_dir, result_xml_dir):
    name_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    anchors_np = np.array([[23., 40.], [36., 84.], [58., 46.], [65., 135.], [111., 89.], [121., 207.], [216., 287.], [228., 138.], [382., 297.]])
    score = 0.3
    iou = 0.45
    model_image_size = (416, 416)
    yolo = YOLO(model_path, name_classes, anchors_np, score, iou, model_image_size)

    if os.path.exists(result_xml_dir):
        shutil.rmtree(result_xml_dir)
    os.makedirs(result_xml_dir)
    if not os.path.exists(test_dir):
        raise ValueError(test_dir, "not exist !")
    img_lists = os.listdir(test_dir)
    img_lists = [i for i in img_lists if i.endswith(".jpg")]
    for img_name in img_lists :
        img_path = os.path.join(test_dir, img_name)
        if not os.path.exists(img_path) :
            raise ValueError(img_path,"not exist !")
        anno = GEN_Annotations(img_name)
        image = Image.open(img_path)
        anno.set_size(image.size[0], image.size[1], 3)  # img shape
        boxes_str = yolo.detect_image(image)
        print("img_path:", img_path, "boxes_str:\n", boxes_str)
        for _,box in enumerate(boxes_str.split()) :
            # print("box",_+1)
            label = box.split(",")[0]
            score = box.split(",")[1]
            xmin = box.split(",")[2]
            ymin = box.split(",")[3]
            xmax = box.split(",")[4]
            ymax = box.split(",")[5]
            anno.add_pic_attr(label, score, xmin, ymin, xmax, ymax)  # label boxes
        anno.savefile(os.path.join(result_xml_dir,img_name.replace(".jpg",".xml")))  # "00001.xml"
    yolo.close_session()

if __name__ == '__main__':
    model_path = 'models_loc/yolo_voc_model.h5'
    test_dir = "gt_voc"
    result_xml_dir = "pd_xml_voc_test"
    model_infer(model_path , test_dir, result_xml_dir)

