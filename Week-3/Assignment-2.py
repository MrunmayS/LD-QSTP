import numpy as np 
import cv2
import matplotlib.pyplot as plt

imgpath = 'E:\LD-QSTP\Week-3\img_05.jpg'
img =  cv2.imread(imgpath)
img_og = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (21,21),cv2.BORDER_DEFAULT)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2,150, param1 =50, param2=20, minRadius=220, maxRadius=250 )
circles_int = np.array(circles, dtype='int')
print('No. of coins:',circles_int.shape[1])
#print(circles_int)
centres = circles_int[:1,:,:-1]
centres = centres.reshape((circles_int.shape[1],2))
centres1 = np.copy(centres)
print(circles_int)


for i in range(0,circles_int.shape[1]-1):
    for j in range(1,circles_int.shape[1]):
        if i ==j:
            continue
        a = np.array(list(centres[i]))
        b = np.array(list(centres1[j]))
        dist = np.linalg.norm(centres[i]-centres1[j])
        c = (a + b)/2
        c = c.astype(int)
        #print(c)
        #print(a,b)
        cv2.line(img_og, tuple(centres[i]),tuple(centres[j]),(0,255,0),10)
        cv2.putText(img_og,str(format(dist, "10.4E")),tuple(c),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),5)
        #print(dist)

for i in circles_int[0, :]:
    diameter = 2*i[2]
    cv2.putText(img_og,str(format(diameter, "10.1E")),(i[0],i[1]),cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255),5)
    #cv2.circle(img_og, (i[0],i[1]), i[2], (0,255,0),5)

plt.imshow(img_og)
plt.show()
