import numpy as np
import cv2

src = cv2.imread('C:\\Users\\whj\\Pictures\\Saved Pictures\\CV.jpg',cv2.IMREAD_UNCHANGED)
src = cv2.resize(src,(1368,720))
cv2.namedWindow('白色风景',cv2.WINDOW_NORMAL)
