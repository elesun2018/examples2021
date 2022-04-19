"""
pyautogui是比较简单的，但是不能指定获取程序的窗口，因此窗口也不能遮挡，不过可以指定截屏的位置，0.04s一张截图，比PyQt稍慢一点，但也很快了。
https://blog.csdn.net/jokerzhanglin/article/details/117201541
https://blog.csdn.net/up1292/article/details/103629712
https://blog.csdn.net/apollo_miracle/article/details/103947116
"""
import pyautogui
import cv2

import pyautogui
import cv2
 
 
img = pyautogui.screenshot() # 截取整个屏幕
img = pyautogui.screenshot(region=[0,0,100,100]) # 截取一个区域 region参数x,y,w,h
img.save('screenshot-04.png')
# img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

# 识别图片, 未成功识别,返回None; 成功识别,返回首次发现该图像时左边的x,y坐标,宽度和高度
logo_path = "logo.png"
pos_loc = pyautogui.locateOnScreen(logo_path)
# 灰度值匹配 把grayscale参数设置为True来加速定位 默认为False
# pos_loc = pyautogui.locateOnScreen(logo_path, grayscale=True)
print("pos_loc",pos_loc)
if pos_loc is not None :
    # 返回该区域中心的x,y坐标
    xy = pyautogui.center(pos_loc)
    print("type(xy)",type(xy))
    print("xy",xy)

pos_loc = pyautogui.locateCenterOnScreen(logo_path)
print("pos_loc",pos_loc)
#if pos_loc is not None :
    #print("x=%d,y=%d"%(x,y))

#像素匹配
x=10
y=10
#获取屏幕截图中像素的RGB颜色
pix = img.getpixel((x, y))
pix = pyautogui.pixel(x, y)
# 验证单个像素是否与给定像素匹配
flag = pyautogui.pixelMatchesColor(x, y, (130, 135, 144))
print("flag",flag)
#可选的tolerance关键字参数指定在仍然匹配的情况下，红、绿、蓝三个值的变化幅度
flag = pyautogui.pixelMatchesColor(x, y, (130, 135, 144), tolerance=200)
print("flag",flag)