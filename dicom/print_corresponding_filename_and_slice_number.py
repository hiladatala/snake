import numpy as np
import os
import pydicom

paths = [
    '/media/pihash/DATA21/Research/Michael_G/CT_EXPERIMENTS/pytorch_results/perceptualNet_std_low_vs_std_high_soft_hyper_column_loss_vgg/61_std',
    '/media/pihash/DATA21/Research/Michael_G/CT_EXPERIMENTS/pytorch_results/vnet_unet_mul_std_denoised_vs_cs_labels_with_heart_map_bceLoss_map/61_std',
    '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/low_dose/61_std',
    '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/high_dose/61_std'
    ]

d = {}

for path in paths:

    d[path] = []

    for f in os.listdir(path):

        ds = pydicom.read_file(os.path.join(path, f), force = True)

        if not hasattr(ds, 'InstanceNumber'):
            continue
        d[path].append((f, ds.InstanceNumber))

for p in d:
    print(f'path {p}\n')

    for v in d[p]:
        print(v)
