from image.my_image import myImage
from my_time import my_time
import os
import matplotlib.pyplot as plt
from dicom.mask_CT import find_mask
import numpy as np

t = my_time()

baseStr = '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/FBP/high_dose_FBP/'
baseStrOut = '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/sinograms_from_FBP'
os.makedirs(baseStrOut, exist_ok = True)

cases = os.listdir(baseStr)

for i,caseNum in enumerate(cases[:1]):

    caseFullPath = os.path.join(baseStr, caseNum)
    print('creating sinogram for {} ...'.format(caseNum))
    pathOut = os.path.join(baseStrOut, caseNum)
    image = myImage(caseFullPath)
    t.tic()
    sinogram = image.create_sinogram_3D_fan_beam()
    backProjected = image.create_backprojection(100)
    backProjected[backProjected > image.I[100,...].max()] = 0
    M = find_mask(image.I[100,...])
    backProjected = backProjected * M
    print('sum(abs) is : {}'.format(np.mean(np.abs(backProjected - (image.I[100,...] * M)))/backProjected.size))
    f, axes = plt.subplots(2)
    axes[0].imshow(backProjected, cmap='gray')
    axes[1].imshow(image.I[100,...] * M, cmap='gray')
    plt.show()
    s = t.toc(ret = True)
    print('finished creating sinogram for {}'.format(caseNum))
    print('time took : ' + s)
    # image.write_sinogram(im = sinogram, path = pathOut, source_path = caseFullPath, description = 'Sinogram from FBP')
