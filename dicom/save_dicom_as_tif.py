import numpy as np
from image.my_image import myImage
import os
import matplotlib.pyplot as plt
import imageio
from skimage.util import invert
from PIL import Image

dataType = 'tiff'
basePath = []
basePath.append('/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/low_dose')
basePath.append('/media/pihash/DATA2/Research/Michael_G/CT_EXPERIMENTS/pytorch_results/perceptualNet_bp_low_vs_bp_high_hyper_column_for_edith')
basePath.append('/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/high_dose')
outPathBase = '/media/pihash/DATA2/Research/Michael_G/CT_EXPERIMENTS/CT_as_{}/Three_CT_nodule'.format(dataType)
cases = ['66']
offset = 8
slices1 = np.arange(70,100)
slices2 = slices1 + offset

for i,d in enumerate(cases):

    I = np.zeros((len(slices1), 512, 512*len(basePath)), dtype = 'int16')
    os.makedirs(os.path.join(outPathBase, d) ,exist_ok = True)

    for k,modality in enumerate(basePath):

        im = myImage(os.path.join(modality, d))
        if 'high_dose' in modality:
            I[:,:,512*k:512*(k+1)] = im.I[slices2,:,:].astype('int16')
        else:
            I[:,:,512*k:512*(k+1)] = im.I[slices1,:,:].astype('int16')

    I = I - I.min()

    imageio.mimwrite(os.path.join(outPathBase,d ,'DD.{}'.format(dataType)), I.astype('float32'))

    # for s in range(I.shape[0]):
    #     print('writing {}'.format(os.path.join(outPathBase,d ,str(s) + '.{}'.format(dataType))))
    #     imageio.imwrite(os.path.join(outPathBase,d ,str(s) + '.{}'.format(dataType)), I[s,...])
