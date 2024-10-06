import matplotlib.pyplot as plt
from skimage import data,color,morphology, io,filters
#生成二值测试图像
img = io.imread('./2.jpg')
val = filters.threshold_otsu(img)
g = img < val
#g=color.rgb2gray(img)
#g=(g<0.5)*1

#R = img[:,:,0]
#G = img[:,:,1]
#B = img[:,:,2]

#g = R*g+G*g+B*g



chull = morphology.convex_hull_image(g)

#绘制轮廓
fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1= axes.ravel()
ax0.imshow(g,plt.cm.gray)
ax0.set_title('original image')

ax1.imshow(chull,plt.cm.gray)
ax1.set_title('convex_hull image')

plt.show()
