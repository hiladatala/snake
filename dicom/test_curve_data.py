import numpy as np
from PIL import ImageDraw, Image
import array
import matplotlib.pyplot as plt
import os
import pydicom


def read_curve_data(ds, fill=True):
    k = 0
    mask = []

    while (0x5000 + (k * 2), 0x3000) in ds:
        print(f'extracting curve_data from slice {ds.InstanceNumber}')
        curve_data_string = ds[0x5000 + (k * 2), 0x3000].value
        curve_data = np.array(array.array('f', curve_data_string))
        roi = curve_data
        x_data = roi[0::2]
        y_data = roi[1::2]
        v = np.round(np.vstack((x_data, y_data))).astype(int)
        mask.append(Image.new('L', ds.pixel_array.shape, 0))
        xy = [val for pair in zip(v[0, :], v[1, :]) for val in pair]
        ImageDraw.Draw(mask[k]).polygon(xy, outline=1, fill=fill)
        k += 1

    if len(mask):

        mask = [np.array(m) for m in mask]
        mask = np.sum(mask, axis=0)
        mask[mask > 0] = 1
        mask = mask.astype(int)

    else:

        mask = None

    return mask


dicomDir = 'PATH_TO_DICOM_DIRECTORY'
for f in os.listdir(dicomDir):

    ds = pydicom.read_file(os.path.join(dicomDir, f),
                           force=True)
    if not hasattr(ds, 'InstanceNumber'):
        continue

    if ds is None:
        continue

    mask = read_curve_data(ds)
    if mask is not None:
        plt.figure()
        plt.imshow(mask, cmap='gray')
        plt.title(f'slice {ds.InstanceNumber}')

plt.show()
