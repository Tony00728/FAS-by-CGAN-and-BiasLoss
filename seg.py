import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data, filter, segmentation, measure, morphology, color,io

#生成二值测试图像
img = io.imread('./2.jpg')
img1 = io.imread('./2.jpg')
#img=color.rgb2gray(img)
#img=(img<0.5)*1
image = color.rgb2gray(img)
thresh =filter.threshold_otsu(image)
bw =morphology.closing(image < thresh, morphology.square(14))

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

#R = color.gray2rgb(R)

img[:,:,0] = bw*R
img[:,:,1] = bw*G
img[:,:,2] = bw*B

img1 = img1-img

#X = bw*R+bw*G+bw*B

io.imshow(img1)
plt.show()
