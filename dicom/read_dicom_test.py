import os
from dicom.my_dicom import myDicom
import matplotlib.pyplot as plt

basePath = '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/Enriched_study/all_cases/high_dose/26_std'
dicom = myDicom()

I, meta = dicom.readDicomSeriesWithMeta(basePath)
# I = dicom.readDicomSeries(basePath)

plt.figure()
plt.imshow(I[30], cmap='gray', clim=(100, 500))
plt.figure()
plt.imshow(I[30], cmap='gray', clim=(800, 1300))
plt.show()
