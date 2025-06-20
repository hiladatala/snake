from my_image import myImage
import os
from tqdm import tqdm

basePath = '/media/pihash/503CA32E7D15980E/reut'
outPath = f'/media/pihash/794A-619A/Fetal_MRI'
im = myImage()
ds = tqdm(os.listdir(basePath))

for d in ds:
    im.read_im(os.path.join(basePath, d), convertFromPacs = False)
    im.write_image(path = os.path.join(outPath,d), anonymize = True)
