FROM img.ai.ebiot.net/seeraishow/face_detect:v1.0.1
# 文件及文件夹目录拷贝
COPY demo /opt/demo
#COPY demo/dist/server-detectFace /opt/demo
WORKDIR /opt/demo
RUN chmod +x /opt/demo -R

EXPOSE 8053
# 配置启动命令
# ENTRYPOINT python3 server-detectFace.py >log 2>&1
# ENTRYPOINT server-detectFace >log 2>&1
