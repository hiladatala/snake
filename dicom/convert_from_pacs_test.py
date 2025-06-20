import os
from dicom.my_dicom import myDicom

basePath = '/media/pihash/DATA2/MAMO/real_data'
dicom = myDicom()
dicom.convertFromPacs = True

for Dir in os.listdir(basePath):
    print('converting {}'.format(os.path.join(basePath, Dir)))
    dicom.convertFolderFromPACS(os.path.join(basePath, Dir))