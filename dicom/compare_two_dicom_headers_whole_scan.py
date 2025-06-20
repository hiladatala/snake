from pathlib2 import Path
from compare_two_dicom_headers import main


if __name__ == '__main__':
    base1 = Path('/media/pihash/DATA/Research/Michael_G/Liver/experiment_scans_sorted/26085068/501_Ax_OP/IP_BH')
    base2 = Path('/media/pihash/DATA/Research/Michael_G/Liver/experiment_scans_sorted/27278605/501_Ax_OP/IP_BH/')
    numSlices = len(list(base1.iterdir()))
    slices = range(numSlices)
    slices = list(slices)[10:11]

    for S in slices:
        main(base1=base1,
             base2=base2,
             sliceNum1=S,
             sliceNum2=S)
