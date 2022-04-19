FROM ubuntu:18.04
#FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get -qq update && \
    apt-get install -yq \
        unzip \
        vim \
        # python3.6 \
        # python3-pip \
        # 解决opencv导入问题
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-dev && \
    apt-get clean && \
    apt-get autoclean -q && \
    apt-get autoremove -q && \
    rm -rf /var/lib/apt/lists/*
# 设置容器内pip下载源
#COPY pip.conf /root/.pip/pip.conf
# 在线安装-基础包
#RUN pip3 --no-cache-dir --quiet install -i https://mirrors.aliyun.com/pypi/simple \
#        --upgrade pip \
#        numpy \
#        pandas \
#        opencv-python==4.1.2.30
#        scikit-learn
# 离线包安装方式-上面安装不成功，推荐
# COPY torch-1.7.0+cpu-cp36-cp36m-linux_x86_64.whl /utils/
# RUN pip3 --no-cache-dir --quiet install /utils/torch-1.7.0+cpu-cp36-cp36m-linux_x86_64.whl
# COPY torchvision-0.8.1+cpu-cp36-cp36m-linux_x86_64.whl /utils/
# RUN pip3 --no-cache-dir --quiet install /utils/torchvision-0.8.1+cpu-cp36-cp36m-linux_x86_64.whl
# 在线安装-包文档
#COPY requirements.txt /utils/
#RUN pip3 --no-cache-dir --quiet install -i https://mirrors.aliyun.com/pypi/simple -r /utils/requirements.txt
#RUN rm -rf /utils
# 文件及文件夹目录拷贝
COPY demo/dist /opt/demo
WORKDIR /opt/demo
RUN chmod +x /opt/demo -R

EXPOSE 8060
# 配置启动命令
ENTRYPOINT ./server-opencv >log 2>&1
