from image.my_image import myImage
from my_time import my_time
import os

t = my_time()

IOs = [50e4]
base_str = '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/high_dose/'
casesForTraining_ = ['22_std', '30_std', '32_std']

for IO in IOs:

    base_str_out = '/media/pihash/DATA/Research/Michael_G/CT_EXPERIMENTS/syn_low_dose_lungs_' + str(int(IO)) + '/'
    casesForTraining = casesForTraining_

    if casesForTraining is None:
        casesForTraining = os.listdir(base_str)

    casesForTraining = [base_str + x for x in casesForTraining]

    if(not os.path.exists(base_str_out)):
        os.mkdir(base_str_out)

    for Case in casesForTraining:

        print(f'creating synthetic for case {Case} with IO = {IO}' )
        patientNum = Case.split('/')[-1]
        pathOut = base_str_out + patientNum
        image = myImage(Case)
        t.tic()
        image.add_poisson_noise(IO = IO)
        s = t.toc(ret = True)
        print('finished creating synthetic for ' + Case)
        print('time took : ' + s)
        print('writing result')
        image.write_image_(im = image.noisy, path = pathOut, source_path = Case, description = f'NON diagnostic low dose poisson - {IO}')
