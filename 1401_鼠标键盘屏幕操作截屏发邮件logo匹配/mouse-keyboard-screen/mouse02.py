"""
python控制鼠标
实时显示鼠标位置
https://www.cnpython.com/qa/78280
"""
import os,time,pyautogui

while True:
    os.system('CLS') # DOS/Windows
    # os.system('clear') # Unix/Linux/MacOS/BSD/etc
    x,y = pyautogui.position()
    print('鼠标位置：x=%04d,y=%04d'%(x,y))
    time.sleep(1)


  



