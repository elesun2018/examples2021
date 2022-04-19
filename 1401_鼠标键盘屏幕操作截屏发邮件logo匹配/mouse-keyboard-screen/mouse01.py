"""
Python直接控制鼠标
https://www.cnblogs.com/fanghao/p/8453207.html
"""
import os
import pyautogui

screenWidth, screenHeight = pyautogui.size() # 屏幕尺寸
mouseX, mouseY = pyautogui.position() # 返回当前鼠标位置，注意坐标系统中左上方是(0, 0)

pyautogui.PAUSE = 1.5 # 每个函数执行后停顿1.5秒
pyautogui.FAILSAFE = True # 鼠标移到左上角会触发FailSafeException，因此快速移动鼠标到左上角也可以停止

w, h = pyautogui.size()
pyautogui.moveTo(w/2, h/2) # 基本移动
pyautogui.moveTo(100, 200, duration=2) # 移动过程持续2s完成
#pyautogui.moveTo(None, 500) # X方向不变，Y方向移动到500

pyautogui.moveRel(-40, 500) # 相对位置移动


# 点击+向下拖动
pyautogui.click(941, 34, button='left')
pyautogui.dragRel(0, 100, button='left', duration=5)
# 点击
pyautogui.click(300, 400, button='right') # 包含了move的点击，右键
pyautogui.click(clicks=2, interval=0.25) # 双击，间隔0.25s
# 滚轮
pyautogui.scroll(-10)
