import numpy as np 
import cv2
import matplotlib.pyplot as plt

imgpath = 'E:\LD-QSTP\Week-3\img_02.jpg'
img =  cv2.imread(imgpath)
img_og = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (21,21),cv2.BORDER_DEFAULT)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2,100, param1 =50, param2=30, minRadius=220, maxRadius=250 )
circles_int = np.array(circles, dtype='int')
print('No. of circles:',circles_int.shape[1])
print(circles_int)
centres = circles_int[:1,:,:-1]
centres = centres.reshape((2,2))
print(centres)
cv2.line(img_og, tuple(centres[0]),tuple(centres[1]),(0,255,0),5)
count = 0 
"""
for i in centres[0, :]:
    count = count + 1
    cv2.line(img_og, tuple(i), tuple(centres[count]), (0,255,0),5)
    print(i)

"""
plt.imshow(img_og)
plt.show()
