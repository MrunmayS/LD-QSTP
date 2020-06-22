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

def pie_chart(hist, centroids):
	
	sizes  = list()
	explode = (0.01, 0.01, 0.01, 0.01, 0.01)
	colors = list()
	cnt = 0 
	a = np.array([1])
	for (percent, color) in zip(hist, centroids):
		sizes.append((percent * 360))
		r,g,b = color
		r = r/255
		b = b/255
		g = g/255
		color = np.array([r,g,b])
		if cnt == 0:
			colors = np.hstack((color,a))
		cnt = cnt + 1
		color1 = np.hstack((color,a))
		colors = np.vstack((colors,color1))
		#colors = np.hstack((colors,a))
	
	pie = plt.pie(sizes, explode = explode, colors = colors)
	#color_array = np.array(colors)
	#print(colors)


img = cv2.imread("E:\LD-QSTP\Week-3\img_01.png")
#print(img)
h,w,l = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# For percent
img1 = img.reshape((h*w,l))
clt1 = KMeans(n_clusters= 5)
clt1.fit(img1)
hist = find_hist(clt1)
#img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
#print(img_rgba.shape)
bar = plot_colors(hist,clt1.cluster_centers_)
#pie_c = pie_chart(hist, clt1.cluster_centers_)


# For Quant

img2 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
img2 = img2.reshape((h*w,l))
clt2 = KMeans(n_clusters=5)
labels = clt2.fit_predict(img2)
quant = clt2.cluster_centers_.astype("uint8")[labels]

#img = img.reshape((h,w,l))
quant = quant.reshape((h,w,l))
#img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)

plt.axis("off")
#plt.subplot(121)
plt.imshow(quant)
plt.show()
#plt.subplot(122)
plt.axis("off")
plt.imshow(bar)
plt.show()

plt.axis("off")
pie_chart(hist, clt1.cluster_centers_)
plt.show()