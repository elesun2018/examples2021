"""
功能：
python捕捉屏幕，并查看屏幕中的任何移动。屏幕始终被捕获并更新。
https://www.cnpython.com/qa/333039
"""
import numpy as np
import cv2
from PIL import ImageGrab

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    cv2.imshow('window', screen_bgr)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

