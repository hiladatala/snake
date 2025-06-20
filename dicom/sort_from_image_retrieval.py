import numpy as np
from image.my_image import myImage
import os
from collections import defaultdict
import shutil
import traceback
from tqdm import tqdm

basePath = '/media/pihash/DATA21/abdomen/highBMI_renal_colic_org'
outPath = f'{basePath}_sorted'
im = myImage()
os.makedirs(outPath, exist_ok = True)
accessions = defaultdict(int)
plane = 'Axial'
tagToSort = None #{'tag' : ,'value' : }

dirs = tqdm(os.listdir(basePath))
for d in dirs:

    try:

        accession = d.split('_')[0]
        pathTodir = os.path.join(basePath, d)
        im.read_im(pathTodir, only_first_dcm = True, dcm_idx = 1)

        if im.plane is None or im.plane == 'undefined':
            continue

        if not plane is None and im.plane != plane:
            continue

        if not tagToSort is None and str(im.meta[0][tagToSort['tag']].value) != str(tagToSort['value']):
            continue

        dirs.set_description(f'processing case {d}')

        accessions[accession] += 1
        pathToDirOut = os.path.join(outPath, accession, str(accessions[accession]))
        shutil.copytree(src = pathTodir, dst = pathToDirOut)

    except Exception as e:
        print(f'{traceback.format_exc()}')

for accession in accessions:

    if accessions[accession] == 1:
        baseDir = os.path.join(outPath, accession)
        dirToRemove = os.path.join(baseDir, str(accessions[accession]))
        print(f'removing dir {dirToRemove}')
        for F in os.listdir(dirToRemove):
            shutil.copyfile(src = os.path.join(dirToRemove, F), dst = os.path.join(baseDir, F))
        shutil.rmtree(dirToRemove)





