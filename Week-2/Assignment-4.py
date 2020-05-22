import cv2
import matplotlib.pyplot as plt
import numpy as np 

imgpath = 'Week-2\img.PNG'
img =  cv2.imread(imgpath)
img_og = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (21,21),cv2.BORDER_DEFAULT)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2,100, param1 =50, param2=30, minRadius=50, maxRadius=200 )
circles_int = np.array(circles, dtype='int')
print('No. of circles:',circles_int.shape[1])

for i in circles_int[0, :]:
    cv2.circle(img_og, (i[0],i[1]), i[2], (0,255,0),5)
plt.imshow(img_og)
plt.show()