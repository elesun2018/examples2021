# -*- coding: utf-8 -*-
'''
多机多卡分布式训练样例
利用 Keras 来训练多工作器（worker）
https://blog.csdn.net/weiyuxuan123/article/details/107938724
https://tensorflow.google.cn/tutorials/distribute/multi_worker_with_keras?hl=zh_cn
https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/distribute/multi_worker_with_keras.ipynb
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from tensorflow import keras  #tensorflow-gpu==2.0.0
import tensorflow as tf
import json
from tensorflow.keras.datasets import mnist

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["10.1.69.34:10162", "10.1.68.150:10163"]
    },
    'task': {'type': 'worker', 'index': 1}
})

def process_data(data_path, img_rows, img_cols, channels, num_classes):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data(data_path)
    print("x_test load_data",x_test)
    print("y_test load_data",y_test)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("y_test to_categorical",y_test)
    return (x_train, y_train), (x_test, y_test)

def build_model(img_rows, img_cols, channels, num_classes):
    input_shape = (img_rows, img_cols, channels)
        
    inputs = keras.Input(shape=(img_rows, img_cols, channels))
    x = keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape)(inputs)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    #model = keras.models.Sequential()
    #model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #model.add(keras.layers.Dropout(0.25))
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(128, activation='relu'))
    #model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    model.summary()
    return model



num_classes = 10
epochs = 10
batch_size = 32
# input image dimensions
img_rows, img_cols ,channels= 28, 28, 1

#(x_train, y_train), (x_test, y_test) = process_data("mnist.npz",img_rows, img_cols, channels, num_classes)
x_train = np.random.random((60000, img_rows, img_cols, channels)) # 生成 行x列的随机浮点数，浮点数范围 : (0,1)
y_train = np.random.random((60000, num_classes))
x_test = np.random.random((10000, img_rows, img_cols, channels))
y_test = np.random.random((10000, num_classes))
print("x_train.shape:",x_train.shape)
print("y_train.shape:",y_train.shape)
print("x_test.shape:",x_test.shape)
print("y_test.shape:",y_test.shape)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = build_model(img_rows, img_cols, channels, num_classes)
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_test, y_test))         


