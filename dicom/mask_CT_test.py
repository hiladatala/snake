from dicom.mask_CT import find_mask
from image.my_image import myImage
import matplotlib.pyplot as plt

path = '/media/pihash/DATA2/Research/Michael_G/CT_EXPERIMENTS/pytorch_results/CHEN_24/24'
im = myImage(path).I

sliceToCheck = 100
mask = find_mask(im)
maskedIm = mask * im

fig, ax = plt.subplots(2,1)
ax[0].imshow(mask[sliceToCheck, :, :], cmap = 'gray')
ax[1].imshow(maskedIm[sliceToCheck, :, :], cmap = 'gray')
plt.show()


