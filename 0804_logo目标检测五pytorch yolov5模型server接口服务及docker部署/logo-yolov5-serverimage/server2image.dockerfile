#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get -qq update && \
    apt-get install -yq \
        unzip \
        vim \
        python3.6 \
        python3-pip \
        # 解决opencv导入问题
        libsm6 \
        libxrender1 \
        libxext6 && \
    apt-get clean && \
    apt-get autoclean -q && \
    apt-get autoremove -q && \
    rm -rf /var/lib/apt/lists/*
# 设置容器内pip下载源
COPY pip.conf /root/.pip/pip.conf
# 在线安装-基础包
RUN pip3 --no-cache-dir --quiet install -i https://mirrors.aliyun.com/pypi/simple \
        --upgrade pip \
        numpy \
        pandas \
        matplotlib \
        Pillow \
        opencv-python==4.1.2.30 \
        # 安装tf-gpu 注意版本对应关系
#        tensorflow==2.1.0 \
#        keras==2.3.1 \
        # 安装torch-gpu 注意版本对应关系
        torch==1.6.0 \
        torchvision==0.7.0 \
        scikit-learn
# 离线包安装方式-上面安装不成功，推荐
#COPY xxx.whl /utils/
#RUN pip3 --no-cache-dir --quiet install /utils/xxx.whl
# 在线安装-包文档
COPY requirements.txt /utils/
RUN pip3 --no-cache-dir --quiet install -i https://mirrors.aliyun.com/pypi/simple -r /utils/requirements.txt
RUN rm -rf /utils
# 文件及文件夹目录拷贝
COPY demo /opt/demo
WORKDIR /opt/demo
RUN chmod +x /opt/demo -R

EXPOSE 4056
# 配置启动命令
ENTRYPOINT python3 server.py >log 2>&1
