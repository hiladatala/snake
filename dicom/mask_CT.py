# from medpy.filter import largest_connected_component
from scipy.ndimage import binary_fill_holes
import copy
import numpy as np
from scipy.ndimage import binary_opening

def find_mask(im, Th = None):

    if(Th is None):
        Th = 500
    if(len(im.shape) > 2):
        slices = np.arange(start = 0, stop = im.shape[0], step = 1)
        MASK = np.zeros(im.shape, dtype = im.dtype)
        for s in slices:

            mask = copy.deepcopy(im[s,:,:])
            mask[mask < Th] = 0
            mask[mask > 0] = 1
            mask = binary_fill_holes(mask)
            mask = largest_connected_component(mask)
            mask = binary_opening(mask, iterations = 15)
            MASK[s,:,:] = mask
    else:

        mask = copy.deepcopy(im)
        mask[mask < Th] = 0
        mask[mask > 0] = 1
        mask = binary_fill_holes(mask)
        mask = largest_connected_component(mask)
        mask = binary_opening(mask, iterations = 15)
        MASK = mask


    return MASK

