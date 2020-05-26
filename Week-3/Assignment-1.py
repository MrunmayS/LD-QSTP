import cv2
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cluster import KMeans

imgpath = "E:\LD-QSTP\Week-3\img_01.png"
img_bgr = cv2.imread(imgpath)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def find_histogram(clt):
    numLabels = np.arange(0,len(np.unique(clt.labels_))+1)
    hist, _ - np.histogram(clt.labels_, bins = numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

img = img.reshape((img.shape[0]*img.shape[1],3))
clt = KMeans(n_clusters=5)
clt.fit(img)

hist1 = find_histogram(clt)
plt.imshow(clt)
plt.show()

"""
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    print(colors[count.argmax()])

unique_count_app(img)
plt.imshow(img)
plt.show()
"""