from pathlib2 import Path
from image.my_image import myImage


def main(base1, base2, sliceNum1, sliceNum2):
    im1 = myImage()
    im2 = myImage()

    im1.read_im(base1.as_posix())
    im2.read_im(base2.as_posix())

    meta1 = im1.meta[sliceNum1]
    meta2 = im2.meta[sliceNum2]

    print(f'Slice1 = {sliceNum1}, Slice2 = {sliceNum2}')
    print(f'{base1} not in {base2}')
    [print(x) for x in tuple([(k.tag, k.name, meta1[k.tag].value) for k in meta1 if k.tag not in meta2])]
    print(f'{base2} not in {base1}')
    [print(x) for x in tuple([(k.tag, k.name, meta2[k.tag].value) for k in meta2 if k.tag not in meta1])]
    print(f'different tag vals')
    [print(x) for x in tuple([(k.tag, k.name, meta1[k.tag].value, meta2[k.tag].value) for
                              k in meta2 if
                              k.tag in meta1 and
                              meta1[k.tag].value != meta2[k.tag].value and
                              k.name != 'Pixel Data'])]


if __name__ == '__main__':
    base1 = Path('/media/pihash/DATA/Research/Michael_G/Liver/MRI/phases/in_phase/15')
    base2 = Path(
        '/media/pihash/DATA2/MRI/cs/10/cs_Ax_T2r5_1')
    sliceNum1 = 6
    sliceNum2 = 6

    main(base1=base1,
         base2=base2,
         sliceNum1=sliceNum1,
         sliceNum2=sliceNum2)
