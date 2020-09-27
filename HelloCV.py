import numpy as np
import cv2

src = cv2.imread('C:\\Users\\whj\\Pictures\\Saved Pictures\\CV.jpg',cv2.IMREAD_UNCHANGED)
src = cv2.resize(src,(1368,720))
#cv2.namedWindow('white',cv2.WINDOW_NORMAL)
cv2.imshow('white', src)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()