import matplotlib.pyplot as plt
from my_image import myImage
import numpy as np
from scipy.ndimage.morphology import binary_closing as fill

im = myImage()
im.read_im('/home/pihash/Pictures/images.jpeg')
from skimage import measure

blobs = im.I > im.I.mean()
all_labels = measure.label(blobs)
blobs_labels = measure.label(blobs, background=0).astype(np.float32)
plt.figure()
plt.imshow(blobs_labels)
plt.title('before')
blobs_labels = fill(blobs_labels).astype(np.float32)
plt.figure()
plt.imshow(blobs_labels)
plt.title('after')
plt.show()