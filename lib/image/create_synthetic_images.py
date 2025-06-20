from image.my_image import myImage
import os

base_str = '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/low_dose/'
base_str_out = '/media/pihash/DATA2/Research/Michael_G/CT_EXPERIMENTS/syn_low_dose'
casesForTraining = ['22']
casesForTraining = [os.path.join(base_str, x) for x in casesForTraining]

os.makedirs(base_str_out, exist_ok = True)

for Case in casesForTraining:
    print('creating synthetic for ' + Case + ' ...')
    patientNum = Case.split('/')[-1]
    pathOut = base_str_out + patientNum
    image = myImage(Case)
    image.add_gaussian_noise(add_to_sinogram = False, sigma = 2)
    print('finished creating synthetic for ' + Case)
    image.write_image_(im = image.noisy, path = pathOut, source_path = Case)
