import sys
sys.path.append('/media/pihash/DATA/Research/Michael_G/python_codes/CT/dicom')
from my_dicom import myDicom
import copy

def reverse_dcm(dicomDir):
    print(f'reversing dir {dicomDir}')
    dicom = myDicom()
    I, meta = dicom.readDicomSeriesWithMeta(dicomDir)
    dicom.writeDicomSeries(I = I[::-1,:,:], source_path = dicomDir, output_path = dicomDir, description = meta[0].SeriesDescription)

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        caseToReverse = sys.argv[1]
        caseToReverse = caseToReverse.split(',')
        for Case in caseToReverse:
            reverse_dcm(Case)

