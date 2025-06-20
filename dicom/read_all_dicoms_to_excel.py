import os
import sys
import pandas as pd
import pydicom
from collections import defaultdict
import numpy as np
from pathlib2 import Path
import datetime


def find_dicomdir(base):
    isDicomDir, ds = is_dicomdir(Path(base))
    if isDicomDir:
        return isDicomDir, ds
    dirs = list(Path(base).iterdir())
    dicomDir = None
    for d in dirs:
        isDicomDir, ds = is_dicomdir(d)
        if isDicomDir:
            dicomDir = d
            break

    return dicomDir, ds


def fix_dates(_cols):
    dateKeys = [x for x in list(cols.keys())
                if 'date' in x.lower()]
    for dateKey in dateKeys:
        cols[dateKey] = [datetime.datetime
                         .strptime(x[:4] + ':' + x[4:6] + ':' + x[6:],
                                   '%Y:%m:%d') for x in _cols[dateKey]]
    return cols


def is_dicomdir(Dir):
    dicom_dir = False
    ds = None
    if not Dir.is_dir():
        return dicom_dir, ds
    for File in Dir.iterdir():
        try:
            ds = pydicom.read_file(File.as_posix(), force=True)
            if hasattr(ds, 'AccessionNumber'):
                dicom_dir = True
                break
        except Exception as e:
            print(e)

    return dicom_dir, ds


def remove_redundant_cols(cols):
    lens = [(x, len(cols[x])) for x in cols]
    maxLen = max([x[1] for x in lens])
    keysToSave = [x[0] for x in lens if x[1] == maxLen]
    cols = {k: cols[k] for k in keysToSave}
    return cols


excelFileName_ = 'all_enriched_low.xlsx'
basePath = Path('/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/Enriched_study/excels')
excelFileName = basePath.joinpath(excelFileName_).as_posix()
basePaths = ['/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/Enriched_study/low_dose']

tags = [(0x10, 0x20),
        (0x8, 0x22),
        (0x20, 0x11),
        (0x8, 0x50),
        (0x18, 0x1210),
        (0x18, 0x60)]
tagToSort = (0x8, 0x22)
# if the sorted values are ints use sortInts
sortInts = True
# if we want to add data from multiple scans to the same row choose a tag that will change,
# otherwise choose tagToChange = None
tagToChange = (0x20, 0x11)
cols = defaultdict(list)

dirsToTake = None

if tagToChange is not None:
    tags = [x for x in tags if x != tagToChange]

tagNames = []

for basePath in basePaths:

    for d in os.listdir(basePath):

        if dirsToTake is not None and d not in dirsToTake:
            continue

        if not os.path.isdir(os.path.join(basePath, d)):
            continue

        dicomDir, ds = find_dicomdir(os.path.join(basePath, d))
        if dicomDir is None:
            continue

        if isinstance(tagToSort, tuple):
            tagToSort = ds[tagToSort].name

        tagName = ds[tagToChange].name + '_' + os.path.split(basePath)[1]
        tagName = tagName if len(d.split('_')) == 1 else tagName + '_' + d.split('_')[-1]

        if ds[tags[0]].value in cols[ds[tags[0]].name]:
            idTag = cols[ds[tags[0]].name].index(ds[tags[0]].value)
            if idTag > len(cols[tagName]) - 1:
                cols[tagName] += [''] * (idTag - len(cols[tagName]) + 1)
            cols[tagName][idTag] = ds[tagToChange].value
        else:
            for tag in tags:
                cols[ds[tag].name].append(ds[tag].value)
            cols[tagName].append(ds[tagToChange].value)
            cols['case#'].append(d)

cols = remove_redundant_cols(cols)
cols = fix_dates(cols)
df = pd.DataFrame(data=dict(cols))
if sortInts:
    newIndex = np.argsort([cols[tagToSort][x] for x in range(len(cols[tagToSort]))])
    df = df.reindex(newIndex)
else:
    df.sort_values(tagToSort, inplace=True)
df.to_excel(excelFileName, index=None)

print('excel saved in ' + excelFileName)
