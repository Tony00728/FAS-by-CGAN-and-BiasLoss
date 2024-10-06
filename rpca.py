"""
==============
Normalized Cut
==============

This example constructs a Region Adjacency Graph (RAG) and recursively performs
a Normalized Cut on it [1]_.

References
----------
.. [1] Shi, J.; Malik, J., "Normalized cuts and image segmentation",
       Pattern Analysis and Machine Intelligence,
       IEEE Transactions on, vol. 22, no. 8, pp. 888-905, August 2000.
"""

from skimage import data, segmentation, color, io, filters
from skimage.future import graph
from matplotlib import pyplot as plt


#img = data.coffee()
img = io.imread('./2.jpg')

labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1, mode='similarity')
#labels2 = graph.cut_normalized(labels1, g)
#out2 = color.label2rgb(labels2, img, kind='avg')
#out2 = out2-(out2/abs(out2))
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
out2 = val = filters.threshold_otsu(img)
mask = img < val
#ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.tight_layout()

plt.tight_layout()
plt.show()