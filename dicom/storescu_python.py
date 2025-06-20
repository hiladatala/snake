import os
import sys

def upload_dir_to_PACS(directory):
    print('uploading dicom folder ' + directory + ' to the PACS')
    os.system('D:/storescu_Michael.bat ' + directory)

if(len(sys.argv) == 1):
    raise RuntimeError('No input argument!')

if(sys.argv[1] == '-h'):
    print('Usage:\n-dir --> directory of multiple dicom cases that is required to be uploaded to the PACS')
    print('-case --> one directory of a specific case that is required to be uploaded to the PACS')

if(sys.argv[1] == '-case'):
    caseDir = sys.argv[2]
    upload_dir_to_PACS(caseDir)

if(sys.argv[1] == '-dir'):
    casesDir = sys.argv[2]
    dirs = os.listdir(casesDir)
    for caseDir in dirs:
        upload_dir_to_PACS(caseDir)