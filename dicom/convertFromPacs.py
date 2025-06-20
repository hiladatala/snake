import os
from my_dicom import myDicom

dicom = myDicom()
s = '/media/pihash/4A5E5D055E5CEAE9/MRI_PROSTATE'

for d in os.listdir(s):
    try:
        dicom.convertFromPACS(os.path.join(s,d))
    except Exception as e:
        print('problem with {}'.format(e))