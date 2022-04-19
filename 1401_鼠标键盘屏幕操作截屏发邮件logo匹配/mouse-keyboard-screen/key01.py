"""
Python
https://blog.csdn.net/ebzxw/article/details/80740515
https://www.jb51.net/article/180619.htm
https://blog.csdn.net/guangmingsky/article/details/80009547
https://blog.csdn.net/weixin_43430036/article/details/84650938

"""
import os,time
import pyautogui

time.sleep(5)
pyautogui.typewrite('Hello python')
pyautogui.press('enter') # 接受按键命令

pyautogui.typewrite(message='Hello world!',interval=0.5)
pyautogui.press('enter') # 接受按键命令

pyautogui.keyDown('shift');pyautogui.press('4');pyautogui.keyUp('shift') # 输出 $ 符号的按键
pyautogui.press('enter') # 接受按键命令

pyautogui.hotkey('ctrl','a') #组合键
pyautogui.hotkey('ctrl','c') #组合键
pyautogui.press('down')  # press the left arrow key
pyautogui.press('enter') # 接受按键命令
pyautogui.hotkey('ctrl','v') #组合键
pyautogui.press('enter') # 接受按键命令
#pyautogui.keyDown('ctrl');pyautogui.keyDown('v');pyautogui.keyUp('v');pyautogui.keyUp('ctrl') #热键组合

# press() ：键盘功能按键
pyautogui.press('enter') # press the Enter key
pyautogui.press('f1')   # press the F1 key
pyautogui.press('left')  # press the left arrow key
pyautogui.press('esc')







