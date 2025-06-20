from dicom.my_dicom import myDicom
import os

def reverse_dcm(baseDir):

    dcm = myDicom()

    dirs = [os.path.join(baseDir,x) for x in os.listdir(baseDir)]

    for D in dirs:

        try:

            J, meta = dcm.readDicomSeriesWithMeta(D)
            dcm.writeDicomSeries(I = J[::-1,:,:], source_path = D, output_path = D, description = meta[0].SeriesDescription)
            print('reversing case num ' + D)

        except Exception as e:
            print('invalid case ' + D +  ' error: ' + str(e))

if __name__ == '__main__':

    cases = ['32', '44', '48', '58', '60']
    baseDirs = ['/media/pihash/DATA2/Research/Michael_G/CT_EXPERIMENTS/registration_results/{}_rigid_moving_is_low'.format(x) for x in cases]
    for baseDir in baseDirs:
        reverse_dcm(baseDir)
