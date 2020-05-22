import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)   
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
imgbgr = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
imginv= cv2.bitwise_not(img)
imghsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
imghsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.figure()


plt.subplot(231)
plt.title("RGB")
plt.xticks([])
plt.yticks([])
plt.imshow(img)

plt.subplot(232)
plt.title("Inverted")
plt.xticks([])
plt.yticks([])
plt.imshow(imginv)

plt.subplot(233)
plt.title("BGR")
plt.xticks([])
plt.yticks([])
plt.imshow(imgbgr)

plt.subplot(234)
plt.title("HSV")
plt.xticks([])
plt.yticks([])
plt.imshow(imghsv)

plt.subplot(235)
plt.title("HSL")
plt.xticks([])
plt.yticks([])
plt.imshow(imghsl)
plt.show()
cap.release()