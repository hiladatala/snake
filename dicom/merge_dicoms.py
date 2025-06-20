import numpy as np
from image.my_image import myImage
import os
import matplotlib.pyplot as plt
import imageio
from skimage.util import invert

dataType = 'tiff'
basePath = []
basePath.append('/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/low_dose')
basePath.append('/media/pihash/DATA2/Research/Michael_G/CT_EXPERIMENTS/pytorch_results/perceptualNet_bp_low_vs_bp_high_hyper_column_for_edith')
basePath.append('/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/high_dose')
Case = '66'
outPathBase = f'/media/pihash/DATA2/Research/Michael_G/CT_EXPERIMENTS/merged_dicom/Three_CT_nodule/{Case}'
offset = 8
slices1 = np.arange(70,100)
slices2 = slices1 + offset

d = Case
I = np.zeros((len(slices1), 512, 512*len(basePath)), dtype = 'int16')
os.makedirs(os.path.join(outPathBase, d) ,exist_ok = True)

for k,modality in enumerate(basePath):

    im = myImage(os.path.join(modality, d))
    if 'high_dose' in modality:
        I[:,:,512*k:512*(k+1)] = im.I[slices2,:,:].astype('int16')
    else:
        I[:,:,512*k:512*(k+1)] = im.I[slices1,:,:].astype('int16')

im.set_image(I)
metaValToChange = [{'tagID' : (0x28,0x1050), 'value' : -400},{'tagID' : (0x28,0x1051), 'value' : 1500}]
im.write_image(path = outPathBase, description = 'ULD (left), ULD denoised by Voxellence (center), Normal-Dose (right)', anonymize = True, metaValToChange = metaValToChange)

