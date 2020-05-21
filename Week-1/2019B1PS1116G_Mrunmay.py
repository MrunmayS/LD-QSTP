import numpy as np
import matplotlib as plt
from PIL import Image
path = input("Specify img path:")
img = Image.open(path)
nimg = np.array(img)
invarr = np.invert(nimg)
invimg = Image.fromarray(invarr)
invpath = input("Specify inverted path:")
invimg.save(invpath)
rgbwts = np.array([0.2989, 0.5870, 0.1140])
monoarr = np.dot(nimg,rgbwts)
monoimg = Image.fromarray(monoarr)
#monopath = input("Specify monochrome path:")
#monoimg.save(monopath)
monoimg.show()

