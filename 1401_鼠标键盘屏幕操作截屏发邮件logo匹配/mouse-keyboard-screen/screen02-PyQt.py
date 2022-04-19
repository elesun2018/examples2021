"""
1.通过win32gui模块，调用windows系统的截屏功能，对屏幕进行录制。
2.通过timer定时器，实现每隔2秒钟，截屏一次，从而记录屏幕使用者的操作记录。
3.对截取的屏幕按照 截取时间进行命名，并存储到文件路径中。
https://m.jb51.net/article/213910.htm
"""


from PyQt5.QtWidgets import QApplication
import win32gui
import sys
import time
record = win32gui.FindWindow(None, 'C:\Windows\system32\cmd.exe')
app = QApplication(sys.argv)
def timer(n):
    while True:  
        dt= time.strftime('%Y-%m-%d %H%M%S',time.localtime())
        screen = QApplication.primaryScreen()
        img = screen.grabWindow(record).toImage()
        img.save(dt+".jpg")
        time.sleep(n)
if __name__ == "__main__":
    timer(2)
 

