import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_hist(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    #print(hist)
    return hist

def plot_colors(hist, centroids):
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	return bar

img = cv2.imread("E:\LD-QSTP\Week-3\img_01.png")
h,w,l = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((h*w,l))

clt = KMeans(n_clusters= 5)
clt.fit(img)
hist = find_hist(clt)
bar = plot_colors(hist,clt.cluster_centers_)
img = img.reshape((h,w,l))
img1 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

img1 = img1.reshape((h*w,l))
labels = clt.fit_predict(img1)
quant = clt.cluster_centers_.astype("uint8")[labels]

img = img.reshape((h,w,l))
quant = quant.reshape((h,w,l))
img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
plt.axis("off")
plt.subplot(121)
plt.imshow(quant)

plt.subplot(122)
plt.axis("off")
plt.imshow(bar)
plt.show()

