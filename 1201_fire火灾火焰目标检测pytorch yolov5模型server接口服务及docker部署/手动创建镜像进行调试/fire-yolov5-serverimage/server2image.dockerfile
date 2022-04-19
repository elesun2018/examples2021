FROM fireyolov5-conserver2image-sfz:0.1

# 文件及文件夹目录拷贝
COPY demo /opt/demo
WORKDIR /opt/demo
RUN chmod +x /opt/demo -R

EXPOSE 9730 22
# 配置启动命令
ENTRYPOINT python3 server-fire.py >log 2>&1
