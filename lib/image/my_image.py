import pathlib
import time
from os.path import exists
import numpy as np
import pathlib2
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from skimage import io
from dicom.my_dicom import myDicom
import joblib
import matplotlib.pyplot as plt
import nibabel as nib
# import cv2
import random
# import av
import pydicom
from matplotlib import path
import scipy.misc
from skimage.util import view_as_windows as viewW
from matplotlib.widgets import Slider, Button, RadioButtons
import copy
# import monai
# import astra
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
import scipy.ndimage as ndimage
# from PIL import Image
from matplotlib.widgets import RectangleSelector, LassoSelector, EllipseSelector
import matplotlib.patches as patches_
from dicom.mask_CT import find_mask
from scipy.ndimage import affine_transform as Affine
from scipy.ndimage.filters import gaussian_filter as blur
from skimage.exposure import equalize_hist as histeq
import os
from matplotlib.widgets import Slider, Button, RadioButtons
import inspect
import array
from PIL import ImageDraw
from skimage.morphology import binary_opening, dilation, disk, erosion
# from medpy.filter import largest_connected_component
from scipy.ndimage.morphology import binary_fill_holes as fill_foles
import imageio
from collections import defaultdict
from skimage.draw import ellipse as get_ellipse_inds
from skimage.feature import corner_harris, corner_peaks
from matplotlib.patches import Ellipse
from pathlib2 import Path


class IndexTracker(object):
    def __init__(self, ax, X, clim=[None, None], cmap='gray'):

        self.ax = ax
        self.X = X
        rows, cols, self.slices = X.shape[:3]
        self.ind = 0
        # self.ind = 15

        self.im = ax.imshow(self.X[:, :, self.ind], cmap=cmap, vmin=clim[0], vmax=clim[1])
        self.update()

    def onscroll(self, event):

        if getattr(event, 'key', None) == 'enter':
            self.saved_pos = self.ind
            plt.close('all')
            return
        elif getattr(event, 'button', None) == 'up' or getattr(event, 'key', None) == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class myImage:
    nameToRemove = ''

    def __init__(self, im=None,
                 name=None,
                 cmap=None,
                 clim=None,
                 mask=None,
                 maskTh=None,
                 affineTransform=None,
                 useFSIM=False,
                 insertInstanceNumber=False,
                 convertToRange=None,
                 dtype=None,
                 transforms=None,
                 numDims=None,
                 filterMeta=None):

        self.dicom = myDicom()
        self.dtype = dtype
        self.maxSlicesTwoShow = 10
        self.showNoisy = False
        self.transforms = transforms
        self.convertToRange = convertToRange
        self.pos = None
        self.cmap = cmap
        self.affine = affineTransform
        self.mask = mask
        self.interpolation = None
        self.maskTh = maskTh
        self.maskedImage = False
        self.clim = clim
        self.insertInstanceNumber = insertInstanceNumber
        self.conversionMethod = 'func'
        self.funcName = None
        # self.cropForgroundFunc = monai.transforms.CropForeground(select_fn=lambda x: x > 300)
        # self.cropForgroundFunc2 = monai.transforms.CropForeground(select_fn=lambda x: x > 0,
        #                                                           return_coords=True)
        self.ROIS = []
        self.reconstructedPath = None

        if self.transforms is not None:
            if not isinstance(self.transforms, list):
                self.transforms = [self.transforms]
            # if len(self.transforms):
            #     self.transforms = monai.transforms.Compose(self.transforms)
            # else:
            #     self.transforms = None
            self.transforms = None

        if (useFSIM):
            from for_articles.FSIM import FSIM
            self.FSIM = FSIM
        if (not im is None):
            self.set_image(im=im, name=name, numDims=numDims, filterMeta=filterMeta)

    def set_image(self, im, name=None, slices=None, numDims=None, filterMeta=None):

        if isinstance(im, Path):
            im = im.as_posix()

        if (isinstance(im, str)):
            self.read_im(im, slices=slices, filterMeta=filterMeta)
        else:
            self.I = im
            if self.affine is not None:
                self.I = Affine(self.I, np.linalg.inv(self.affine))
            if slices is not None:
                self.I = self.I[slices, :, :]

            if isinstance(im, list):
                self.imShape = im[0].shape
            else:
                self.imShape = im.shape

            if (name is None):
                self.imName = 'image'
            else:
                self.imName = name
            if (not self.mask is None and self.mask):
                M = find_mask(im=self.I, Th=self.maskTh)
                self.I = self.I * M
            self.convert_to_range()

        if self.I is None:
            return
        if (not self.dtype is None):
            self.I = self.I.astype(dtype=self.dtype)

        if not numDims is None:
            dimsToAdd = numDims - self.I.ndim
            if dimsToAdd > 0:
                dimsToAdd = tuple(np.arange(dimsToAdd))
                self.I = np.expand_dims(self.I, *dimsToAdd)

    def __getitem__(self, item):

        return self.I.__getitem__(item)

    def astype(self, Type):
        return self.I.astype(Type)

    @property
    def ndim(self):
        if hasattr(self, 'I'):
            return self.I.ndim
        else:
            return tuple([])

    @property
    def shape(self):
        if hasattr(self, 'I'):
            return self.I.shape
        else:
            return tuple([])

    @property
    def plane(self):
        return self.dicom.plane(self.meta)

    def reverse(self):

        self.I = self.I[::-1, ...]

    def read_all_metas_in_dir(self, path):

        dirs = os.listdir(path)
        meta = []

        for d in dirs:
            self.read_im(os.path.join(path, d), only_first_dcm=True)
            meta.append(self.meta[0])

        self.meta = meta

    def read_im(self, im, dtype=None,
                slices=None,
                name=None,
                convertToRange=None,
                nonZeroSlices=False,
                reverse=False,
                convertFromPacs=False,
                filterMeta=None,
                return_hu=False,
                force_dicom=False,
                printProcess=True,
                **kwargs):

        if isinstance(im, pathlib2.Path) \
                or isinstance(im, pathlib.Path):
            im = im.as_posix()

        self.dicom.convertFromPacs = convertFromPacs
        insertInstanceNumber = kwargs.get('insertInstanceNumber', None)
        self.insertInstanceNumber = self.insertInstanceNumber \
            if insertInstanceNumber is None else insertInstanceNumber
        if 'insertInstanceNumber' in kwargs:
            del kwargs['insertInstanceNumber']

        if dtype is not None:
            self.dtype = dtype
        if convertToRange is not None:
            self.convertToRange = convertToRange

        if isinstance(im, list):
            self.I = [self.read_im(x,
                                   dtype,
                                   slices,
                                   name,
                                   convertToRange,
                                   nonZeroSlices,
                                   convertFromPacs) for x in im]
            return self.I

        if isinstance(im, str):

            if os.path.isdir(im) and len(os.listdir(im)) == 1:
                im = os.path.join(im, os.listdir(im)[0])

            self.sourcePath = im

            if im in ['jpg', 'png', 'JPG', 'jpeg', 'JPEG']:
                self.I = ndimage.imread(im)
            elif 'tiff' in im:
                self.I = np.array(imageio.mimread(im))
            elif 'nii' in im:
                self.I = nib.load(im)
                self.I = nib.as_closest_canonical(self.I).get_fdata()
                self.I = np.transpose(self.I, (2, 1, 0))[:, ::-1, ::-1]
            else:
                inputArgs = inspect.getfullargspec(self.dicom.readDicomSeriesWithMeta).args
                args = {key: kwargs[key] for key in inputArgs if key in kwargs}
                self.I, self.meta = self.dicom.readDicomSeriesWithMeta(im,
                                                                       insertInstanceNumber=self.insertInstanceNumber,
                                                                       return_hu=return_hu,
                                                                       useForceRead=force_dicom,
                                                                       printProcess=printProcess,
                                                                       **args)
                # delete pixel array from meta
                # for idx in range(len(self.meta)):
                #     del self.meta[idx][(0x7FE0, 0x10)]
                self.sliceNumsFilenames = self.dicom.sliceNumsFilenames
                if isinstance(self.I, list):

                    rows = [x.shape[-2] for x in self.I]
                    cols = [x.shape[-1] for x in self.I]
                    self.imShape = (len(self.I), np.max(rows), np.max(cols))
                    I_ = self.I
                    self.I = np.zeros(self.imShape, dtype=I_[0].dtype)

                    for s in range(len(I_)):
                        self.I[s, :I_[s].shape[-2], :I_[s].shape[-1]] = I_[s]

            if name is None:
                self.imName = im.split('/')[-1]
            else:
                self.imName = name
        else:
            self.I = im
            if name is None:
                self.imName = 'image'
            else:
                self.imName = name

        if self.affine is not None:
            self.I = Affine(self.I, np.linalg.inv(self.affine))

        self.orgShape = self.I.shape
        if slices is not None:
            self.orgShape = copy.deepcopy(self.I.shape)
            if not isinstance(slices, np.ndarray):
                slices = np.array(slices)
            if slices.max() >= self.I.shape[0]:
                self.I = np.concatenate([self.I] + [np.zeros((slices.max() -
                                                              self.I.shape[0] + 1,
                                                              self.I.shape[1],
                                                              self.I.shape[2]))])
            self.I = self.I[slices, :, :]
        self.imShape = self.I.shape

        if self.mask is not None and self.mask:
            if not hasattr(self, 'M'):
                self.M = find_mask(im=self.I, Th=self.maskTh)
            self.I = self.I * self.M

        self.convert_to_range()

        if self.dtype is not None:
            self.I = self.I.astype(dtype=self.dtype)

        if nonZeroSlices:
            nzSlices = np.any(self.I, axis=(1, 2))
            self.I = self.I[nzSlices, ...]

        self.dicom.convertFromPacs = False

        if filterMeta is not None:
            self.filter_by_meta(filterMeta)

        if 'valToTake' in kwargs \
                and str(kwargs.get('valToTake')) != 'all':
            valToTake = kwargs.get('valToTake')
            if isinstance('valToTake', str):
                valToTake = int(valToTake)
            self.I[self.I != valToTake] = 0

        if reverse:
            self.I = self.I[::-1]

        return self.I

    def calc_hist(self, im=None,
                  show=True,
                  bins=None,
                  sliceNum=None, title=None,
                  width=0.8,
                  Th=None,
                  noFig=False):

        if (im is None):
            im = self.I
        if Th is None:
            Th = im[sliceNum, ...].min() - 1 if not sliceNum is None else im.min() - 1
        if (bins is None):
            bins = int(im.max())
        if (not sliceNum is None):
            self.hist = np.histogram(im[sliceNum, ...][im[sliceNum, ...] > Th], bins=bins)
        else:
            self.hist = np.histogram(im[im > Th], bins=bins)
        if not noFig:
            plt.figure()
            plt.bar(self.hist[1][:-1], self.hist[0], width=width)
            plt.title(title)
        if show:
            plt.show()

        return self.hist

    def imshow_3d(self, *args, **kwargs):

        '''

        :param args: multiple 3D images separated with commas , e.g., im1,im2
        each image should be of size NxMxZ
        :param kwargs:
        title = ['title1','title2'] (optional),
        'blockSize' = [ysize,xsize,zsize] (optional)
        :return: None
        '''

        args = list(args)

        if not len(args):
            args.append(self.I)

        if 'blockSize' in kwargs and len(kwargs['blockSize']) == 3:

            for k in range(len(args)):
                args[k] = np.concatenate(args[k], axis=0)

        self.fig3d = plt.figure()
        if 'title' in kwargs:
            titles = kwargs['title']
        if 'clim' in kwargs:
            clim = kwargs['clim']
        else:
            clim = [None, None]

        nrows = np.ceil(len(args) / 5).astype('int')
        ncols = np.ceil(len(args) / nrows).astype('int')
        subplotIndex = 1
        ax = []
        trackers = []
        cmap = kwargs.get('cmap', 'gray')

        for i in range(len(args)):

            ax.append(self.fig3d.add_subplot(nrows, ncols, subplotIndex))
            if 'title' in kwargs:
                if len(titles) == 1:
                    ax[i].set_title(titles[0])
                else:
                    ax[i].set_title(titles[i])

            subplotIndex += 1
            if args[i].ndim == 3:
                im = args[i].transpose(1, 2, 0)
            elif args[i].ndim == 4:
                im = args[i].transpose(1, 2, 0, 3)
            else:
                im = args[i][..., None]

            trackers.append(IndexTracker(ax[i], im, clim=clim, cmap=cmap))
            self.fig3d.canvas.mpl_connect('scroll_event', trackers[i].onscroll)
            self.fig3d.canvas.mpl_connect('key_press_event', trackers[i].onscroll)

        plt.show()

        has_saved_pos = [x for x in trackers if hasattr(x, 'saved_pos')]
        if len(has_saved_pos):
            self.saved_pos_from_im3d = getattr(has_saved_pos[0], 'saved_pos')
        else:
            self.saved_pos_from_im3d = None

    def is_rectangle(self, im):

        is_rec = False
        rectParams = None

        coords = corner_peaks(corner_harris(im), min_distance=3, threshold_rel=0.02)

        if coords.shape == (4, 2):
            if coords[1, 1] - coords[0, 1] == coords[-1, 1] - coords[-2, 1] and coords[1, 0] - coords[0, 0] == coords[
                -1, 0] - coords[-2, 0]:
                is_rec = True
                rectParams = (
                    *tuple(coords[0, ::-1]), coords[1, 1] - coords[0, 1], coords[-1, 0] - coords[0, 0])  # x,y,w,h

        return is_rec, rectParams

    def read_lesion_contours_from_PACS(self, idx, mask):

        lesionData_ = self.read_lesion_data_PACS(sliceNum=idx)

        if len(lesionData_):

            is_rec_list = [self.is_rectangle(im) for im in mask]
            recParams = [x[1] for x in is_rec_list if not x[1] is None]
            is_rec_list = [x[0] for x in is_rec_list]

            lesionMasks = []
            maskInd = 0
            for indContour in range(len(mask)):
                if not is_rec_list[indContour]:
                    lesionMasks.append([])
                    lesionMasks[maskInd].append(mask[indContour])
                else:
                    if len(lesionMasks) < maskInd + 1:
                        lesionMasks.append([mask[indContour]])
                    lesionMasks[maskInd] = np.sum(lesionMasks[maskInd], axis=0)
                    lesionMasks[maskInd][lesionMasks[maskInd] > 0] = 1
                    maskInd += 1

            for indLes, les in enumerate(lesionData_):
                if len(recParams):
                    les['rect'] = recParams[indLes]
                les['mask'] = lesionMasks[indLes]
                lesionData_[indLes] = les

            if any(is_rec_list):
                mask = np.array(mask)[~np.array(is_rec_list), ...]

        return mask, lesionData_

    def read_lesion_data_PACS(self, sliceNum=0):

        # TODO add code for the case of multiple lesions in a slice

        tag = (0x6003, 0x1090)
        vals = []

        if tag in self.meta[sliceNum]:

            vals_ = self.meta[sliceNum][tag].value

            if isinstance(vals_, tuple) or isinstance(vals_, list) or isinstance(vals_, dict):

                vals.append({x.split(':')[0]: x.split(':')[1].replace(' ', '') for x in vals_})
                for v in ['Mean', 'Max', 'Radius']:
                    if not v in vals:
                        continue
                    vals[-1][v] = float(vals[-1][v].replace('cm', ''))

        return vals

    def read_text(self):

        txt = []
        txtTag = (0x6003, 0x1090)

        for header in self.meta:

            if header is None:
                continue

            txt_ = None
            if txtTag in header:
                txt_ = header[txtTag].value
            txt.append(txt_)

        return txt

    def crop_to_data(self):

        mip = self.I.max(axis=0)
        ys = np.where(mip.sum(axis=1) > 0)[0]
        y1, y2 = (ys[0], ys[-1])
        xs = np.where(mip.sum(axis=0) > 0)[0]
        x1, x2 = (xs[0], xs[-1])

        self.x1y1 = [x1, y1]
        self.set_image(im=self.I[:, y1:y2, x1:x2])

        if hasattr(self, 'maskLes'):
            self.maskLes = self.maskLes[:, y1:y2, x1:x2]

        if hasattr(self, 'lesData'):
            for sliceKey in self.lesData:
                for lesNum in range(len(self.lesData[sliceKey])):
                    self.lesData[sliceKey][lesNum]['mask'] = self.lesData[sliceKey][lesNum]['mask'][y1:y2, x1:x2]

        for metaInd in range(len(self.meta)):
            spacing = self.meta[metaInd][0x28, 0x30].value  # row spacing \ column spacing
            position = self.meta[metaInd][0x20, 0x32].value
            position[0] += self.x1y1[0] * spacing[1]
            position[1] += self.x1y1[1] * spacing[0]
            elem = pydicom.DataElement((0x20, 0x32), self.meta[metaInd][0x20, 0x32].VR, position)
            self.meta[metaInd][0x20, 0x32] = elem

    def read_curve_data(self,
                        fill=True,
                        Th=None,
                        dontPrint=False):

        total_mask = np.zeros(self.I.shape)
        lesionData = defaultdict(list)

        for idx, slice in enumerate(self.meta):

            k = 0
            mask = []

            if slice is None:
                continue

            while (0x5000 + (k * 2), 0x3000) in slice:

                if not dontPrint:
                    print('Slice-{} extracting curve'.format(idx))
                curve_data_string = slice[0x5000 + (k * 2), 0x3000].value
                curve_data = np.array(array.array('f', curve_data_string))
                roi = curve_data
                x_data = roi[0::2]
                y_data = roi[1::2]
                v = np.round(np.vstack((x_data, y_data))).astype(int)
                if hasattr(slice, 'pixel_array'):
                    mask.append(Image.new('L',
                                          slice.pixel_array.T.shape,
                                          0))
                else:
                    mask.append(Image.new('L',
                                          self.I[idx].T.shape,
                                          0))

                xy = [val for pair in zip(v[0, :], v[1, :]) for val in pair]
                if len(xy) > 2:
                    ImageDraw.Draw(mask[k]).polygon(xy, outline=1, fill=fill)
                k += 1

            if len(mask):

                mask = [np.array(m) for m in mask]
                mask, lesionData_ = self.read_lesion_contours_from_PACS(idx=idx, mask=mask)
                lesionData[idx] = lesionData_

                mask = np.sum(mask, axis=0)
                mask[mask > 0] = 1
                total_mask[idx] = mask

                if Th is not None:
                    if hasattr(slice, 'pixel_array'):
                        maskedImage = slice.pixel_array * total_mask[idx]
                    else:
                        maskedImage = self.I[idx] * total_mask[idx]
                    total_mask[idx][maskedImage < Th] = 0

        if len(lesionData):
            return total_mask.astype('int'), lesionData
        else:
            return total_mask.astype(int)

    def convert_to_range(self):

        if (self.convertToRange is None):
            return

        OldMax = self.I.max()
        OldMin = self.I.min()
        NewRange = self.convertToRange[1] - self.convertToRange[0]
        OldRange = (OldMax - OldMin)

        if (OldRange == 0):
            newIm = self.I
        else:
            newIm = (((self.I - OldMin) * NewRange) / OldRange) + self.convertToRange[0]

        self.I = newIm

    @staticmethod
    def curves_callback(dataset, data_element):
        if myImage.nameToRemove in data_element.name:
            del dataset[data_element.tag]

    def remove_private_tags(self):

        for i, meta in enumerate(self.meta):
            self.meta[i].remove_private_tags()

    def remove_meta_with_name(self, name):

        myImage.nameToRemove = name

        for i, meta in enumerate(self.meta):
            meta.walk(myImage.curves_callback)
            self.meta[i] = meta

    def histeq(self, useMask=False, nbinsToCalc=None):

        if not nbinsToCalc is None:
            nbins = nbinsToCalc

        if (self.I.ndim > 2):

            for i in range(self.imShape[0]):

                if nbinsToCalc is None:
                    nbins = self.I[i, ...].max()
                if useMask:
                    pos = self.get_pos(im=self.I[i, ...])
                    mask = np.zeros(self.I[i, ...].shape)
                    mask[int(pos[1]):int(pos[1] + pos[3]), int(pos[0]):int(pos[0] + pos[2])] = 1
                    tmp = histeq(self.I[i, ...], nbins=nbins, mask=mask)
                else:
                    tmp = histeq(self.I[i, ...], nbins=nbins)
                self.I[i, ...] = tmp * self.I[i, ...].max()

        else:

            if nbinsToCalc is None:
                nbins = self.I.max()

            if useMask:
                pos = self.get_pos(im=self.I)
                mask = np.zeros(self.I.shape)
                mask[int(pos[1]):int(pos[1] + pos[3]), int(pos[0]):int(pos[0] + pos[2])] = 1
                tmp = histeq(self.I, nbins=nbins, mask=mask)
            else:
                tmp = histeq(self.I, nbins=nbins)
            self.I = tmp * self.I.max()

    def filter_by_meta(self, metaToFilter):

        slicesToKeep = []
        if not isinstance(metaToFilter, list):
            metaToFilter = [metaToFilter]

        if self.I.ndim < 3:

            flag = True
            for m in metaToFilter:
                if not self.meta[m['tagID']].value == m['value']:
                    flag = False

            if not flag:
                self.I = None
                self.imShape = (0, 0)

        else:

            for i in range(self.imShape[0]):

                flag = True
                for m in metaToFilter:
                    if not self.meta[i][m['tagID']].value == m['value']:
                        flag = False

                if flag:
                    slicesToKeep.append(i)

            self.set_image(im=self.I[slicesToKeep, ...])
            self.meta = [self.meta[s] for s in slicesToKeep]

    def apply_transforms(self):
        if self.transforms is not None:
            self.I = self.transforms(self.I[None])[0].numpy()
            if hasattr(self, 'reconstructed'):
                self.reconstructed = self.transforms(self.reconstructed[None])[0].numpy()

    def apply_mask(self, mask):

        self.maskedImage = True
        if isinstance(mask, str):
            mask = myImage().read_im(mask)
        self.I[mask == 0] = 0
        if hasattr(self, 'reconstructed'):
            self.reconstructed[mask == 0] = 0

    def set_reconstructed(self, im, dtype=None, slices=None):

        self.dtype = dtype
        if isinstance(im, pathlib2.Path) or isinstance(im, pathlib.Path):
            im = im.as_posix()
        if isinstance(im, str):
            self.reconstructedPath = im
            if 'dcm' in im:
                self.reconstructed = self.dicom.readDicom(im)
            else:
                self.reconstructed, self.meta = self.dicom.readDicomSeriesWithMeta(im)
        else:
            self.reconstructed = im
        if self.dtype is not None:
            self.reconstructed = self.reconstructed.astype(dtype=self.dtype)

        if slices is not None:
            self.reconstructed = self.reconstructed[slices, :, :]

        self.set_image(self.I[:, :self.reconstructed.shape[-2], :self.reconstructed.shape[-1]])

        return self.reconstructed

    def sliceNumThreshold(self, Th):
        self.I = self.I[0:Th, :, :]

    def set_noisy(self, noisy):
        self.noisy = noisy

    def clear_ROIS(self):
        self.ROIS = []

    def calc_statistics(self, pos=None, sliceNum=0):

        std = []
        mean = []

        im = self.I if self.I.ndim == 2 else self.I[sliceNum, ...]

        croppedI, pos = self.crop_image(im=im, pos=pos)
        self.pos = pos
        std.append(np.std(croppedI.flatten()))
        mean.append(np.mean(croppedI.flatten()))

        if hasattr(self, 'noisy'):
            noisyIm = self.noisy if self.noisy.ndim == 2 else self.noisy[sliceNum, ...]
            croppedNoisy, __ = self.crop_image(noisyIm, pos)
            std.append(np.std(croppedNoisy.flatten()))
            mean.append(np.mean(croppedNoisy.flatten()))

        return mean, std

    def calc_dose_reduction(self, sliceNum=0, resize=1, pos=None):

        if (len(self.imShape) > 2):
            im = self.I[sliceNum, :, :]
            imNoisy = self.noisy[sliceNum, :, :]
        else:
            im = self.I
            imNoisy = self.noisy
        if (resize != 1):
            im = scipy.misc.imresize(im, 100 * resize)
            imNoisy = scipy.misc.imresize(imNoisy, 100 * resize)
        if (not hasattr(self, 'noisy')):
            self.add_gaussian_noise()
        croppedI, pos = self.crop_image(im=im, pos=pos)
        self.pos = pos
        croppedNoisy, __ = self.crop_image(imNoisy, pos)
        stdI = np.std(croppedI.flatten())
        stdNoisy = np.std(croppedNoisy.flatten())
        noiseIncrease = stdNoisy / stdI
        reduction = 100 * (1 - 1 / (noiseIncrease ** 2))
        print('reduction is {}'.format(reduction))
        return reduction

    def create_backprojection(self, sliceNum, algorithm='FBP'):

        newSino = blur(self.sinogram[sliceNum, ...], sigma=2)
        vol_geom = astra.create_vol_geom(self.imShape[1], self.imShape[2])
        reconstruction_id = astra.data2d.create('-vol', vol_geom)
        cfg = astra.astra_dict('{}_CUDA'.format(algorithm))
        cfg['ReconstructionDataId'] = reconstruction_id
        astra.data2d.store(self.sinogram_id[sliceNum], newSino)
        cfg['ProjectionDataId'] = self.sinogram_id[sliceNum]
        cfg['ProjectorId'] = self.project_id[sliceNum]
        algorithm_id = astra.algorithm.create(cfg)
        astra.algorithm.run(i=algorithm_id, iterations=1)
        reconstruction = astra.data2d.get(reconstruction_id)

        return reconstruction

    def change_image_z_positions(self):
        if isinstance(self.meta, dict):
            spacingZ = self.meta['spacing'][-1]
            numSlices = self.meta['spatial_shape'][-1]
            firstZposition = self.meta['00200032']['Value'][-1]
            zPositions = [firstZposition + (spacingZ*idx)
                          for idx in range(numSlices)]
        else:
            zPositions = [int(x[(0x20, 0x32)].value[-1])
                          for x in self.meta]
        zPositions = np.sort(zPositions)
        if not isinstance(self.meta, dict):
            spacingBetweenSlices = float(self.meta[0][(0x18, 0x88)].value)
            zPositions = [zPositions[0] + spacingBetweenSlices * x for x in range(len(self.meta))]
        return zPositions

    def write_image_(self, im,
                     path,
                     source_path='',
                     description='',
                     metaValToChange=None,
                     anonymize=False,
                     seg=None,
                     **kwargs):
        if kwargs.get('fixZPositions'):
            zPositions = self.change_image_z_positions()
            kwargs['zPositions'] = zPositions
        source_path = '' if source_path is None else source_path
        if isinstance(source_path, pathlib2.Path) \
                or isinstance(source_path, pathlib.Path):
            source_path = source_path.as_posix()
        dtype = 'int16' if self.dtype is None else self.dtype
        if source_path == '' and hasattr(self, 'sourcePath'):
            source_path = self.sourcePath
        # if('avi' in path):
        #     self.write_video(im.astype(np.uint8),path)
        if im.shape[0] == 1 or len(im.shape) < 3:
            path = os.path.join(path, source_path.split('/')[-1])
        if '.tif' in path:
            self.write_image_as_tiff(path=path)
        elif '.dcm' in path:
            if source_path == '':
                raise RuntimeError('No source path given for writing dcm, aborting!')
            self.dicom.writeDicom(path,
                                  source_path,
                                  im,
                                  description,
                                  dtype=dtype,
                                  anonymize=anonymize)
        else:
            if kwargs.get('sourceMeta', None) is not None:
                self.dicom.writeDicomSeriesUsingMeta(I=im,
                                                     output_path=path,
                                                     description=description,
                                                     **kwargs)
            else:
                if source_path == '':
                    raise RuntimeError('No source path given for writing dcm, aborting!')
                self.dicom.writeDicomSeries(output_path=path,
                                            source_path=source_path,
                                            I=im,
                                            description=description,
                                            dtype=dtype,
                                            metaValToChange=metaValToChange,
                                            anonymize=anonymize,
                                            seg=seg,
                                            **kwargs)

    def write_sinogram(self, path, im=None, source_path='', description=''):
        if im is None:
            im = self.sinogram
        self.dicom.writeDicomSeriesOnlyImages(output_path=path, source_path=source_path, I=im, description=description)

    def add_gaussian_noise(self, mu=0, sigma=10, add_to_sinogram=True, sliceNum=None):
        self.noiseModel = 'gaussian'
        self.mu = mu
        self.sigma = sigma
        if (add_to_sinogram):
            if (sliceNum is None):
                self.add_noise_to_3d_sinogram()
            else:
                self.add_noise_to_sinogram(sliceNum)
        else:
            N = np.random.normal(mu, sigma, self.imShape)
            self.noisy = self.I + N

    def add_poisson_noise(self, IO=1e4, add_to_sinogram=True, sliceNum=None):
        self.noiseModel = 'poisson'
        self.IO = IO
        if (add_to_sinogram):
            if (sliceNum is None):
                self.add_noise_to_3d_sinogram()
            else:
                self.add_noise_to_sinogram(sliceNum)
        else:
            # self.noisy = (np.random.poisson(IO * self.I.astype(np.float32)) / IO).astype(self.I.dtype)
            # self.noisy = np.random.poisson(IO * np.exp(-self.I/self.I.max())) / (self.I.max() * IO)
            # self.noisy = self.I + np.random.poisson(IO * np.exp(-self.I.astype(np.float32))).astype(self.I.dtype)
            self.noisy = (np.random.poisson(
                self.I.astype(np.float32) / self.I.max().astype(np.float32) * IO) / IO * self.I.max()).astype(
                self.I.dtype)
            # self.noisy = np.random.poisson(IO*np.exp(-self.I.astype(np.float32)/self.I.max().astype(np.float32)) / IO * self.I.max().astype(np.float32)).astype(self.I.dtype)
            # self.noisy = np.random.poisson(IO*np.exp(-self.I + 1e-8)) # self.I * IO) + self.I #/ IO

    def add_noise_to_sinogram(self, sliceNum):
        tmp = copy.deepcopy(self.I)
        # Create geometries and projector.
        vol_geom = astra.create_vol_geom(self.imShape[1], self.imShape[2])
        angles = np.linspace(0, np.pi, 360, endpoint=False)
        proj_geom = astra.create_proj_geom('parallel', 1., 512, angles)
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        # create sinogram
        sinogram_id, sino = astra.create_sino(data=self.I[sliceNum, :, :], proj_id=projector_id, gpuIndex=0)
        # add noise
        if (self.noiseModel == 'poisson'):
            sinoNoisy = astra.functions.add_noise_to_sino(sinogram_in=sino, I0=self.IO)
        elif (self.noiseModel == 'gaussian'):
            sinoNoisy = self.add_gaussian_noise_(im=sino, mu=self.mu, sigma=self.sigma)
        astra.data2d.store(sinogram_id, sinoNoisy)
        # Create reconstruction.
        reconstruction_id = astra.data2d.create('-vol', vol_geom)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = reconstruction_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = projector_id
        algorithm_id = astra.algorithm.create(cfg)
        astra.algorithm.run(i=algorithm_id, iterations=1)
        reconstruction = astra.data2d.get(reconstruction_id)
        reconstruction[self.I[sliceNum, :, :] == 0] = 0
        reconstruction[reconstruction < 0] = 0
        if (not (hasattr('self', 'noisy'))):
            self.noisy = self.I
        self.noisy[sliceNum, :, :] = reconstruction
        self.I = tmp

    def create_sinogram_slice(self, sliceNum):
        vol_geom = astra.create_vol_geom(self.imShape[1], self.imShape[2])
        angles = np.linspace(0, 2 * np.pi, 2 * 360, endpoint=False)
        sourceDetectorDistance = 950
        sourcePatientDistance = 541
        detectorPatientDistance = sourceDetectorDistance - sourcePatientDistance
        det_width = 1.  # Size of a detector pixel.
        det_count = 1500  # Number of detector pixels.
        source_origin = sourcePatientDistance  # Position of the source.
        origin_det = detectorPatientDistance  # Position of the detector
        proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        # create sinogram
        sinogram_id, sino = astra.create_sino(data=self.I[sliceNum, :, :], proj_id=projector_id, gpuIndex=0)
        return sino, sinogram_id, projector_id

    def create_sinogram_3D_fan_beam(self):

        self.sinogram_id = []
        self.project_id = []
        for sliceNum in range(self.imShape[0]):
            if (not sliceNum):
                self.sinogram, sin_id, proj_id = self.create_sinogram_slice(sliceNum=sliceNum)
                self.sinogram = self.sinogram[None, ...]
            else:
                sinogram_, sin_id, proj_id = self.create_sinogram_slice(sliceNum=sliceNum)
                self.sinogram = np.concatenate((self.sinogram, sinogram_[None, ...]), axis=0)
            self.sinogram_id.append(sin_id)
            self.project_id.append(proj_id)
        return self.sinogram

    def add_noise_to_3d_sinogram(self):

        tmp = copy.deepcopy(self.I)
        IforSino = self.I
        # Create geometries and projector.
        vol_geom = astra.create_vol_geom(IforSino.shape[1], IforSino.shape[2], IforSino.shape[0])
        angles = np.linspace(0, np.pi, 360, endpoint=False)
        pixelSpacing = self.meta[0]['0x28', '0x30'].value
        proj_geom = astra.create_proj_geom('parallel3d', 1., 1., IforSino.shape[0], IforSino.shape[1], angles)
        vol_id = astra.data3d.link('-vol', vol_geom, np.ascontiguousarray(IforSino.astype(np.float32)))
        # create sinogram
        sinogram_id, sino = astra.create_sino3d_gpu(data=vol_id, proj_geom=proj_geom, vol_geom=vol_geom, gpuIndex=0)
        self.sinogram = sino
        # add noise
        if (self.noiseModel == 'poisson'):
            sinoNoisy = astra.functions.add_noise_to_sino(sinogram_in=sino, I0=self.IO)
        elif (self.noiseModel == 'gaussian'):
            sinoNoisy = self.add_gaussian_noise_(im=sino, mu=self.mu, sigma=self.sigma)
        self.noisySinogram = sinoNoisy
        astra.data3d.store(sinogram_id, sinoNoisy)
        # Create reconstruction.
        rec_id = astra.data3d.create('-vol', vol_geom)
        cfg = astra.astra_dict('CGLS3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(i=alg_id, iterations=100)
        reconstruction = astra.data3d.get(rec_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(sinogram_id)
        astra.algorithm.delete(alg_id)
        reconstruction[IforSino == 0] = 0
        reconstruction[reconstruction < 0] = 0
        self.noisy = reconstruction
        self.I = tmp

    def add_gaussian_noise_to_sinogram(self, mu, sigma):

        pass

    def add_gaussian_noise_(self, im, mu, sigma):
        if (len(im.shape) == 2):
            N = np.random.normal(mu, sigma, (im.shape[0], im.shape[1]))
        else:
            N = np.random.normal(mu, sigma, (im.shape[0], im.shape[1], im.shape[2]))
        noisy_ = im + N
        return noisy_

    def create_sinogram_3D(self):
        tmp = copy.deepcopy(self.I)
        IforSino = self.I
        # Create geometries and projector.
        vol_geom = astra.create_vol_geom(IforSino.shape[1], IforSino.shape[2], IforSino.shape[0])
        angles = np.linspace(0, np.pi, 360, endpoint=False)
        proj_geom = astra.create_proj_geom('parallel3d', 1., 1., IforSino.shape[0], IforSino.shape[1], angles)
        vol_id = astra.data3d.link('-vol', vol_geom, np.ascontiguousarray(IforSino.astype(np.float32)))
        # create sinogram
        _, self.sinogram = astra.create_sino3d_gpu(data=vol_id, proj_geom=proj_geom, vol_geom=vol_geom, gpuIndex=0)
        self.I = tmp
        return self.sinogram

    def write_noisy_image(self, path, source_patch='', description=''):
        self.write_image_(self.noisy, path, source_patch, description)

    def write_image_as_tiff(self, path):

        os.makedirs(os.path.split(path)[0], exist_ok=True)

        if not '.tiff' in path:
            path += '.tiff'

        if self.I.ndim == 2:
            self.I = self.I[None, :, :]

        print(f'writing file {path}')
        imageio.mimwrite(path, self.I)

    def write_image(self, path,
                    source_path='',
                    description='',
                    metaValToChange=None,
                    anonymize=False,
                    dtype=None,
                    **kwargs):

        # metaValToChange shoould be a list of dictionaries with the desired tagID and value to change:
        # for example: metaValToChange = [{'tagID' : (0x10,0x10), 'value' : 'Pitter'},
        # {'tagID' : (0x8,0x80), 'value' : 'Tel-Aviv'}]
        # this command changes the name of the patient to Pitter and the institution name to Tel-Aviv

        if dtype is not None:
            self.dtype = dtype
        self.write_image_(self.I,
                          path,
                          source_path,
                          description,
                          metaValToChange,
                          anonymize=anonymize,
                          **kwargs)

    def write_sino_noisy(self, path, source_path='', description=''):
        self.write_image_(self.noisySinogram, path, source_path, description)

    def crop(self, pos=None,
             initial=None,
             useSelfPos=None,
             resize=None,
             posForSlice=False):

        if useSelfPos is not None and useSelfPos:
            pos = self.pos

        self.initial = initial

        if posForSlice:
            self.poses = []

        if isinstance(self.I, list):
            self.I = [self.crop_(x, pos, posForSlice=posForSlice) for x in self.I]
        else:
            self.I = self.crop_(self.I, pos, posForSlice=posForSlice)

        if (not self.dtype is None):
            if isinstance(self.I, list):
                self.I = [x.astype(self.dtype) for x in self.I]
            else:
                self.I = self.I.astype(dtype=self.dtype)

        if (hasattr(self, 'reconstructed')):
            self.reconstructed = self.crop_(self.reconstructed, self.pos)
        if (hasattr(self, 'noisy')):
            self.noisy = self.crop_(self.noisy, self.pos)

        if (not resize is None):
            self.resize(resize)

        if isinstance(self.I, list):
            self.imShape = self.I[0].shape
        else:
            self.imShape = self.I.shape

        return self.I, pos

    def crop_(self, im, pos=None, posForSlice=False):

        if (len(self.imShape) > 2):
            tmp, pos = self.crop_scan(im, pos, initialCrop=self.initial, posForSlice=posForSlice)
        else:
            tmp, pos = self.crop_image(im, pos)
        self.pos = pos

        return tmp

    def im2col(self, I=None, blockSize=(5, 5), stepsize=1, sliceNum=None):
        if (I is None):
            if (sliceNum is None):
                patches = viewW(self.I, (blockSize[0], blockSize[1])).reshape(-1, blockSize[0] * blockSize[1],
                                                                              order='F')[:, ::stepsize]
            else:
                patches = viewW(self.I[sliceNum, :, :], (blockSize[0], blockSize[1])).reshape(-1,
                                                                                              blockSize[0] * blockSize[
                                                                                                  1], order='F')[:,
                          ::stepsize]
        else:
            if (sliceNum is None):
                patches = viewW(I, (blockSize[0], blockSize[1])).reshape(-1, blockSize[0] * blockSize[1], order='F')[:,
                          ::stepsize]
            else:
                patches = viewW(I[sliceNum, :, :], (blockSize[0], blockSize[1])).reshape(-1,
                                                                                         blockSize[0] * blockSize[1],
                                                                                         order='F')[:, ::stepsize]
        return patches

    def im3col(self, I=None, blockSize=(5, 5, 5)):
        if (I is None):
            patches = viewW(self.I.transpose((0, 2, 1)), (blockSize[0], blockSize[1], blockSize[2])).reshape(
                (-1, blockSize[0] * blockSize[1] * blockSize[2]))
        else:
            patches = viewW(I.transpose((0, 2, 1)), (blockSize[0], blockSize[1], blockSize[2])).reshape(
                (-1, blockSize[0] * blockSize[1] * blockSize[2]))
        return patches

    def col2im(self, B, blockSize, imageSize):
        m, n = blockSize
        mm, nn = imageSize
        return B.reshape((mm - m + 1, nn - n + 1), order='F')

    def calc_statistics_with_pos(self, pos, metrices, sliceNum=None, **kwargs):

        if not isinstance(pos, list) and not isinstance(pos, tuple):
            pos = [pos]

        out = defaultdict(list)
        self.sliceNumMetric = sliceNum
        numPixelsPerRoi = []

        for metric in metrices:

            for p in pos:
                m = getattr(self, metric, None)
                imCropped = self.get_im_from_pos(p, sliceNum=sliceNum)
                numPixelsPerRoi.append((imCropped > 0).sum())
                if not m is None:
                    inputArgs = inspect.getfullargspec(m).args
                    args = {key: kwargs[key] for key in inputArgs if key in kwargs}
                    out[metric].append(m(imCropped, **args))

        self.numPixelsPerRoi = numPixelsPerRoi

        return out

    def MEAN_ROI(self, im, dataTh=0):

        return im[im > dataTh].mean()

    def CNR(self, im, dataTh=0):

        forground = (im > dataTh).astype('int')
        dilated = dilation(forground, disk(2))
        backround = (dilated ^ forground.astype('bool')).astype('int')
        forground_ = erosion(forground, disk(1)).astype('int')
        if (forground_ > 0).sum() > 0:
            forground = forground_
        forground = forground.astype('float')
        backround = backround.astype('float')
        backround[backround > 0] = self[self.sliceNumMetric, ...][backround > 0]
        forground[forground > 0] = self[self.sliceNumMetric, ...][forground > 0]
        return (np.mean(forground[forground > 0]) - np.mean(backround[backround > 0])) / np.std(
            backround[backround > 0])

    def COV(self, im):

        return (np.std(im) / np.mean(im)) * 100

    def meanSTD(self, im):

        return np.mean(np.std(im))

    def get_im_from_pos(self, pos, sliceNum=None):

        sliceNum = self.I.shape[0] // 2 if sliceNum is None else sliceNum
        if len(pos) > 4:
            # a 3d selection cubical selection
            x, y, z, w, h, d = np.array(pos).astype('int').ravel()
            return self.I[z:z + d, y:y + h, x:x + w]
        elif len(pos) == 4:
            # a 2d rectangle
            x, y, w, h = np.array(pos).astype('int').ravel()
            if self.I.ndim > 2:
                return self.I[sliceNum, y:y + h, x:x + w]
            else:
                return self.I[y:y + h, x:x + w]
        else:
            # if we are here, we've got an ellipse
            tmpI = np.zeros_like(self.I[sliceNum, ...])
            tmpI[pos[0], pos[1]] = self.I[sliceNum, pos[0], pos[1]]
            return tmpI

    def crop_image(self, im=None, pos=None, cropSelf=False, cmap='gray', resize=None):

        if (im is None):
            im = self.I

        initiaPos = None if self.initial is None else (self.initial[-2], self.initial[-1])

        if (pos is None):
            pos = self.get_pos(im=im, cmap=cmap, initialPos=initiaPos)
            # pos = cv2.selectROI('selectROI', im.astype(np.uint8), fromCenter = False, showCrosshair = False)

        print(f'cropping with pos {pos}')
        if (not hasattr(self, 'initial') or self.initial is None):
            cropped = im[int(pos[1]):int(pos[1] + pos[3]), int(pos[0]):int(pos[0] + pos[2])]
        else:
            if pos[1] + self.initial[3] > im.shape[0]:
                pos[1] = im.shape[0] - self.initial[3] - 1
            if pos[0] + self.initial[2] > im.shape[1]:
                pos[0] = im.shape[1] - self.initial[2] - 1
            cropped = im[int(pos[1]):int(pos[1] + self.initial[3]), int(pos[0]):int(pos[0] + self.initial[2])]
        if (cropSelf):
            self.set_image(cropped)
        if (not resize is None):
            cropped = self.resize_(cropped, resize)
        return cropped, pos

    def crop_scan(self, im=None, pos=None, useSelfPos=False, initialCrop=None, posForSlice=False):

        self.initial = initialCrop

        if (useSelfPos):
            pos = self.pos

        if (im is None):
            im = self.I

        if (len(self.imShape) == 2):
            croppedImage, pos = self.crop_image(im)
            return croppedImage, pos

        else:

            croppedScan = []

            if not pos is None and isinstance(pos[0], list):
                poses = copy.deepcopy(pos)
            else:
                poses = self.imShape[0] * [pos]

            for i in range(self.imShape[0]):

                pos = poses[i] if i == 0 else pos
                if i > 0 and not poses[i] is None:
                    pos = poses[i]

                if (i == 0 and pos is None) or posForSlice:

                    if isinstance(im, list):
                        croppedImage, pos = self.crop_image(im[i])
                    else:
                        croppedImage, pos = self.crop_image(im[i, :, :])
                    self.pos = pos

                    if posForSlice:
                        self.poses.append(pos)
                        if i == 0:
                            self.initial = pos

                elif (not pos is None and i == 0):

                    if isinstance(im, list):
                        croppedImage, __ = self.crop_image(im[i], pos)
                    else:
                        croppedImage, __ = self.crop_image(im[i, :, :], pos)

                else:

                    if isinstance(im, list):

                        croppedImage, __ = self.crop_image(im[i], pos)
                    else:
                        croppedImage, __ = self.crop_image(im[i, :, :], pos)

                croppedScan.append(croppedImage[None, :, :])

        croppedScan = np.concatenate(croppedScan, axis=0)

        return croppedScan, pos

    def get_pos_(self, im=None):
        if (im is None):
            im = self.I
        _, pos = self.crop_image(im)
        return pos

    def show_(self, im=None):
        if (im is None):
            im = self.I
        if (len(im.shape) > 2 and im.shape[2] != 3):
            cnt = 0
            for sliceNum in range(self.startSlice, self.imShape[0]):
                cnt += 1
                if (cnt > self.maxSlicesTwoShow):
                    break
                if (not self.showNoisy):
                    fig = plt.figure()
                    fig.canvas.set_window_title(self.imName)
                    plt.imshow(im[sliceNum, :, :], cmap=self.cmap, clim=self.clim, aspect=self.aspect,
                               interpolation=self.interpolation)
                    plt.title(self.title + ' Slice Number is ' + str(sliceNum))
                else:
                    fig, axes = plt.subplots(2, 1)
                    fig.canvas.set_window_title(self.imName + ' slice number is ' + str(sliceNum) + ' norm SSD = ' +
                                                str(np.sum(np.abs(
                                                    im[sliceNum, :, :] - self.noisy[sliceNum, :, :])) / self.noisy[
                                                                                                        sliceNum, :,
                                                                                                        :].size))
                    axes[0].imshow(im[sliceNum, :, :], cmap=self.cmap, clim=self.clim, interpolation=self.interpolation)
                    axes[0].set_title('image')
                    axes[1].imshow(self.noisy[sliceNum, :, :], cmap=self.cmap, clim=self.clim,
                                   interpolation=self.interpolation)
                    axes[1].set_title('image noisy')
        else:
            if (not self.showNoisy):
                fig = plt.figure()
                fig.canvas.set_window_title(self.imName)
                plt.imshow(im, cmap=self.cmap, clim=self.clim, aspect=self.aspect, interpolation=self.interpolation)
                plt.title(self.title)
            else:
                fig, axes = plt.subplots(2, 1)
                fig.canvas.set_window_title(
                    self.imName + self.title + ' norm SSD = ' + str(np.sum(np.abs(im - self.noisy)) / self.noisy.size))
                axes[0].imshow(im, cmap=self.cmap, clim=self.clim, interpolation=self.interpolation)
                axes[0].set_title('image')
                axes[1].imshow(self.noisy, cmap=self.cmap, clim=self.clim, interpolation=self.interpolation)
                axes[1].set_title('image noisy')

    def show_sino_(self):
        if (len(self.imShape) > 2):
            cnt = 0
            for sliceNum in range(self.startSlice, self.imShape[0]):
                cnt += 1
                if (cnt > self.maxSlicesTwoShow):
                    break
                if (not self.showNoisy):
                    fig = plt.figure()
                    fig.canvas.set_window_title(self.imName)
                    plt.imshow(self.sinogram[sliceNum, :, :], cmap='gray', clim=self.clim)
                    plt.title(self.title + ' Slice Number is ' + str(sliceNum))
                else:
                    fig, axes = plt.subplots(2, 1)
                    fig.canvas.set_window_title(self.imName + ' slice number is ' + str(sliceNum) + ' norm SSD = ' +
                                                str(np.sum(np.abs(
                                                    self.sinogram[sliceNum, :, :] - self.noisySinogram[sliceNum, :,
                                                                                    :])) /
                                                    self.noisySinogram[sliceNum, :, :].size))
                    axes[0].imshow(self.sinogram[sliceNum, :, :], cmap='gray', clim=self.clim)
                    axes[0].set_title('image')
                    axes[1].imshow(self.noisySinogram[sliceNum, :, :], cmap='gray', clim=self.clim)
                    axes[1].set_title('image noisy')
        else:
            if (not self.showNoisy):
                fig = plt.figure()
                fig.canvas.set_window_title(self.imName)
                plt.imshow(self.sinogram, cmap='gray', clim=self.clim)
                plt.title(self.title)
            else:
                fig, axes = plt.subplots(2, 1)
                fig.canvas.set_window_title(self.imName + self.title + ' norm SSD = ' +
                                            str(np.sum(
                                                np.abs(self.sinogram - self.noisySinogram)) / self.noisySinogram.size))
                axes[0].imshow(self.sinogram, cmap='gray', clim=self.clim)
                axes[0].set_title('image')
                axes[1].imshow(self.noisySinogram, cmap='gray', clim=self.clim)
                axes[1].set_title('image noisy')

    def resize_(self, im, scale):
        newIm = Image.fromarray(im)
        newIm = newIm.resize((np.int(im.shape[1] * scale), np.int(im.shape[0] * scale)), Image.BICUBIC)
        newIm = np.asarray(newIm, order='F')

        return newIm

    def resize(self, scale):

        if (len(self.imShape) == 2 or self.imShape[2] == 3):
            self.I = self.resize_(self.I, scale)
        else:
            for sliceNum in range(self.imShape[0]):
                im = self.resize_(self.I[sliceNum, :, :], scale)
                if (sliceNum == 0):
                    newIm = np.zeros((self.imShape[0], im.shape[1], im.shape[2]), dtype=self.I.dtype)
                newIm[sliceNum, :, :] = im
            self.I = newIm
            del newIm

    def show(self, title='', dontShow=False, maxSlicesoShow=None, startSlice=None, closeWithMouseClick=False,
             aspect=None, im=None, save=None, interpolation=None):
        self.aspect = aspect
        self.interpolation = interpolation
        if (not startSlice is None):
            self.startSlice = startSlice
        else:
            self.startSlice = 0
        if (not maxSlicesoShow is None):
            self.maxSlicesTwoShow = maxSlicesoShow
        self.useClim = False
        if (hasattr(self, 'noisy')):
            self.showNoisy = True
        else:
            self.showNoisy = False
        self.title = title
        if (im is None):
            im = self.I
        self.show_(im=im)
        if (not save is None):
            plt.savefig('/media/pihash/DATA/Research/Michael_G/python_codes/CT/FIGS/' + save + '.tif')
        if (closeWithMouseClick):
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()
        if (not dontShow):
            plt.show()
        else:
            plt.draw()

    def show_sino(self, title='', dontShow=False, maxSlicesoShow=None, startSlice=None):
        if (not hasattr(self, 'sinogram')):
            return
        if (not startSlice is None):
            self.startSlice = startSlice
        else:
            self.startSlice = 0
        if (not maxSlicesoShow is None):
            self.maxSlicesTwoShow = maxSlicesoShow
        self.useClim = False
        if (hasattr(self, 'noisy')):
            self.showNoisy = True
        else:
            self.showNoisy = False
        self.title = title
        self.show_sino_()
        if (not dontShow):
            plt.show()

    def show_with_map(self, im=None):

        if (im is None):
            im = self.I

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.25, bottom=0.25)
        min0 = 0
        max0 = 25000

        im1 = ax.imshow(im)
        fig.colorbar(im1)

        axcolor = 'lightgoldenrodyellow'
        axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

        smin = Slider(axmin, 'Min', 0, 30000, valinit=min0)
        smax = Slider(axmax, 'Max', 0, 30000, valinit=max0)

        def update(val):
            im1.set_clim([smin.val, smax.val])
            fig.canvas.draw()

        smin.on_changed(update)
        smax.on_changed(update)

        plt.show()

    def show_ROI(self, pos=None, im=None, dont_show=False, title='ROI', edgecolor='r', save=None, interpolation=None):
        if (pos is None and len(self.ROIS) == 0):
            return
        if (im is None):
            im = self.I
        # Create figure and axes
        fig, ax = plt.subplots(1)
        ax.imshow(im, cmap=self.cmap, clim=self.clim, interpolation=interpolation)
        if not isinstance(edgecolor, list) and len(self.ROIS) > 0:
            edgecolor = [edgecolor] * len(self.ROIS)
        # Create a Rectangle patch
        if pos is None and len(self.ROIS) > 0:
            for i, ROI in enumerate(self.ROIS):
                rect = patches_.Rectangle((ROI[0], ROI[1]), ROI[2], ROI[3], linewidth=3, edgecolor=edgecolor[i],
                                          facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
        elif pos is not None:
            rect = patches_.Rectangle((pos[0], pos[1]), pos[2], pos[3], linewidth=3, edgecolor=edgecolor,
                                      facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.title(title)
        if save is not None:
            plt.savefig('/media/pihash/DATA/Research/Michael_G/python_codes/CT/FIGS/' + save + '.tif')
        if not dont_show:
            plt.show()
        else:
            plt.draw()

    def max(self):
        return self.I.max()

    def min(self):
        return self.I.min()

    def sum(self, axis=None):
        return self.I.sum(axis=axis)

    def _get_patches_after_th_std(self, patchesI,
                                  patchesRecon,
                                  th=200):

        stdsRecon = np.std(patchesRecon, axis=(1, 2))
        indices = stdsRecon > th
        patchesI = patchesI[indices]
        patchesRecon = patchesRecon[indices]

        return patchesI, patchesRecon

    def _get_patches_after_th_ssim(self, patchesI,
                                   patchesRecon,
                                   win_size,
                                   th=0.5,
                                   dataRange=None):

        ssims = [ssim(patchesI[idx],
                      patchesRecon[idx],
                      data_range=dataRange,
                      win_size=win_size)
                 for idx in range(len(patchesI))]
        indices = [idx for idx, _ssim in
                   enumerate(ssims) if _ssim > th]
        patchesI = patchesI[np.array(indices)]
        patchesRecon = patchesRecon[np.array(indices)]
        ssims = [ssims[x] for x in indices]

        return patchesI, patchesRecon, ssims

    def compute_psnr_on_patches_with_th_(self, win_size=7,
                                         dataRange=None,
                                         step=30,
                                         thSTD=200,
                                         th=0.5):

        PSNR = self.compute_metric_on_patches_with_th(win_size=win_size,
                                                      dataRange=dataRange,
                                                      step=step,
                                                      thSTD=thSTD,
                                                      th=th,
                                                      func=psnr)
        return PSNR

    def compute_fsim_on_patches_with_th_(self, win_size=7,
                                         dataRange=None,
                                         step=30,
                                         thSTD=200,
                                         th=0.5):

        from image_similarity_measures.quality_metrics import fsim
        FSIM = self.compute_metric_on_patches_with_th(win_size=win_size,
                                                      dataRange=dataRange,
                                                      step=step,
                                                      thSTD=thSTD,
                                                      th=th,
                                                      func=fsim)
        return FSIM

    def _extract_3d_patches(self, blockSize, step=30):

        blockSize = (1, *blockSize)
        patchesI = viewW(arr_in=self.I,
                         window_shape=blockSize)
        patchesRecon = viewW(arr_in=self.reconstructed,
                             window_shape=blockSize)

        return patchesI, patchesRecon

    def _extract_nz_patches(self, patchesI, patchesRecon):

        nzInds = np.logical_and(np.any(patchesI, axis=(1, 2)),
                                np.any(patchesRecon, axis=(1, 2)))
        patchesI = patchesI[nzInds]
        patchesRecon = patchesRecon[nzInds]
        patchSize = patchesI.shape[1:]
        dataInds = np.sum((patchesRecon > 0).astype('int'),
                          axis=(1, 2)) > np.prod(patchSize) * 0.8
        patchesI = patchesI[dataInds]
        patchesRecon = patchesRecon[dataInds]

        return patchesI, patchesRecon

    def compute_metric_on_patches_with_th(self,
                                          win_size=7,
                                          dataRange=None,
                                          step=30,
                                          func=ssim,
                                          blockSize=(30, 30),
                                          thSTD=200,
                                          th=0.5):

        patchesIAll, patchesReconAll = self._extract_3d_patches(blockSize=blockSize,
                                                                step=step)
        if len(patchesIAll) > 98:
            patchesIAll = patchesIAll[98:102]
            patchesReconAll = patchesReconAll[98:102]
        OUT = []
        loader = tqdm(range(len(patchesIAll)))
        for sliceNum in loader:
            patchesI = patchesIAll[sliceNum].reshape(-1, *blockSize)[::step]
            patchesRecon = patchesReconAll[sliceNum].reshape(-1, *blockSize)[::step]
            patchesI, patchesRecon = self._extract_nz_patches(patchesI, patchesRecon)
            patchesI, patchesRecon, ssims = self._get_patches_after_th_ssim(patchesI,
                                                                            patchesRecon,
                                                                            th=th,
                                                                            win_size=win_size,
                                                                            dataRange=dataRange)
            patchesI, patchesRecon = self._get_patches_after_th_std(patchesI,
                                                                    patchesRecon,
                                                                    th=thSTD)
            loader.set_description(f'compute {func.__name__} with th slice {sliceNum}')
            if func.__name__ != 'structural_similarity':
                inputArgs = inspect.getfullargspec(func).args
                kwargs = {}
                if 'data_range' in inputArgs:
                    kwargs = {'data_range': dataRange}
                if func.__name__ == 'fsim':
                    patchesI = patchesI[..., None]
                    patchesRecon = patchesRecon[..., None]
                ssims = [func(patchesI[idx],
                              patchesRecon[idx],
                              **kwargs)
                         for idx in range(len(patchesI))]

            OUT += ssims

        return np.mean(OUT)

    def compute_ssim_on_patches_with_th_(self, win_size=7,
                                         dataRange=None,
                                         step=30,
                                         thSTD=200,
                                         th=0.5):

        SSIM = self.compute_metric_on_patches_with_th(win_size=win_size,
                                                      dataRange=dataRange,
                                                      step=step,
                                                      thSTD=thSTD,
                                                      th=th,
                                                      func=ssim)
        return SSIM

    def compute_fsim_(self, win_size=7,
                      dataRange=None):

        from image_similarity_measures.quality_metrics import fsim
        FSIM = []
        for sliceNum in range(self.imShape[0]):
            I = self.I[sliceNum, :, :]
            reco = self.reconstructed[sliceNum, :, :]
            if (I > 0).sum() > win_size ** 2 \
                    and (reco > 0).sum() > win_size ** 2:
                if self.maskedImage:
                    I, xy1, xy2 = self.cropForgroundFunc2(I[None])
                    I = I[0].numpy()
                    reco = reco[xy1[0]: xy2[0], xy1[1]: xy2[1]]
                FSIM.append(fsim(I[..., None],
                                 reco[..., None]))

        return np.mean(np.array(FSIM)[~np.isnan(FSIM)])

    def compute_ssim_(self, win_size=7,
                      dataRange=None):

        SSIM = []
        dataRange = self.I.max() if dataRange is None \
            else dataRange
        for sliceNum in range(self.imShape[0]):
            I = self.I[sliceNum, :, :]
            reco = self.reconstructed[sliceNum, :, :]
            if (I > 0).sum() > win_size ** 2 \
                    and (reco > 0).sum() > win_size ** 2:
                if not self.maskedImage:
                    I = self.cropForgroundFunc(I[None])[0].numpy()
                    reco = reco[:I.shape[0], :I.shape[1]]
                _, ssimIm = ssim(I,
                                 reco,
                                 full=True,
                                 data_range=dataRange,
                                 win_size=win_size)
                _ssim = ssimIm[I > 0].mean()
                if not np.any(np.isnan(_ssim)):
                    SSIM.append(_ssim)

        return np.mean(SSIM)

    def crop_to_PS(self, im, ps):
        nearestPadRows = int(im.shape[-2] / ps)
        nearestPadCols = int(im.shape[-1] / ps)
        cropCols = int(im.shape[-1] - nearestPadCols * ps)
        cropRows = int(im.shape[-2] - nearestPadRows * ps)
        newim = im[:, cropRows:, cropCols:]
        return newim

    def compute_lpips_(self, ps, Th, net='alex', model_='net-lin'):

        if not hasattr(self, 'dm'):
            import torch as t
            from PerceptualSimilarity.models import dist_model as dm
            self.t = t
            self.dm = dm
            self.model = self.dm.DistModel()
            self.model.initialize(model=model_, net=net, use_gpu=True, version='0.1')

        im0 = self.t.from_numpy(self.I.astype('float32'))
        im1 = self.t.from_numpy(self.reconstructed.astype('float32'))
        im0 = self.crop_to_PS(im0, ps=ps)
        im1 = self.crop_to_PS(im1, ps=ps)
        im0Array = self.t.reshape(im0, (im0.shape[0], -1, ps, ps))
        im1Array = self.t.reshape(im1, (im1.shape[0], -1, ps, ps))

        if not Th is None:
            idxs = ((im0Array > Th).sum(dim=-1).sum(dim=-1) > 0).squeeze()
            im0Array = im0Array[:, idxs, :, :]
            im1Array = im1Array[:, idxs, :, :]

        out = []

        for sliceNum in range(self.I.shape[0]):
            im0 = im0Array[sliceNum, :, :][:, None, :, :]
            im1 = im1Array[sliceNum, :, :][:, None, :, :]
            im0 = self.t.cat((im0, im0, im0), dim=1)
            im1 = self.t.cat((im1, im1, im1), dim=1)
            im0 = (im0 - im0.min()) / (im0.max() - im0.min())
            im1 = (im1 - im1.min()) / (im1.max() - im1.min())
            d = self.model.forward(im0, im1)
            out.append(d.mean())

        return np.mean(out)

    def compute_L1_(self):

        L1 = []
        for sliceNum in range(self.imShape[0]):
            I = self.I[sliceNum, :, :]
            reco = self.reconstructed[sliceNum, :, :]
            L1_ = np.abs(I[I > 0] - reco[I > 0]).mean()
            L1.append(L1_)

        return np.mean(L1)

    def compute_nrmse_(self):

        NRMSE = []
        for sliceNum in range(self.imShape[0]):
            I = self.I[sliceNum, :, :]
            reco = self.reconstructed[sliceNum, :, :]
            NRMSE.append(nrmse(I[I > 0], reco[I > 0]))

        return np.mean(NRMSE)

    def compute_psnr_(self):

        PSNR = []
        dataRange = self.I.max()
        for sliceNum in range(self.imShape[0]):
            I = self.I[sliceNum, :, :]
            reco = self.reconstructed[sliceNum, :, :]
            if self.maskedImage:
                I, xy1, xy2 = self.cropForgroundFunc2(I[None])
                I = I[0].numpy()
                reco = reco[xy1[0]: xy2[0], xy1[1]: xy2[1]]
            PSNR.append(psnr(I[I > 0], reco[I > 0], data_range=dataRange))

        return np.mean(np.array(PSNR)[~np.isnan(PSNR)])

    def get_loc(self, sliceNum=None, numLocs=None, title=None):
        if ((len(self.imShape) > 2 and self.imShape[2] > 3) or not sliceNum is None):
            plt.imshow(self.I[sliceNum, :, :], clim=self.clim, cmap=self.cmap)
        else:
            plt.imshow(self.I, clim=self.clim, cmap=self.cmap)
        if (not title is None):
            plt.title(title)
        loc = plt.ginput(numLocs)
        plt.close()
        return loc

    def read_parameters_for_fun(self, paramFileName):

        if ('lookup' in paramFileName):
            self.lookUpTableCC = joblib.load(paramFileName)['lookuptableCC']
            self.lookUpTableMLO = joblib.load(paramFileName)['lookuptableMLO']
            self.conversionMethod = 'lookuptable'
        else:
            self.conversionMethod = 'func'
            params = joblib.load(paramFileName)
            for parameterName in params:
                setattr(self, 'func{}'.format(parameterName.replace('parameters', 'Params')), params[parameterName])

    def plot_func(self, paramFileName, MAX=None):

        MAX = self.I.max() if MAX is None else MAX
        self.read_parameters_for_fun(paramFileName=paramFileName)
        xplot = np.linspace(start=0, stop=MAX, num=100000)
        plt.plot(xplot, self.func(xplot))
        plt.show()

    def transform_im_with_func(self, paramFileName='', funcName=None):

        if paramFileName != '':
            self.read_parameters_for_fun(paramFileName=paramFileName)
        self.funcName = funcName
        dtype = self.I.dtype
        I = np.zeros(self.imShape, dtype=self.I.dtype)
        for s in range(self.imShape[0]):
            viewPosition = self.meta[s]['0018', '5101'].value
            I[s, ...] = self.func(self.I[s, ...].astype(np.float64), viewPosition)
        # I[I < 0] = 0
        self.I = I.astype(dtype)

    def get_meta_values(self, tagID, names=None):

        metaVals = {}
        imShape = self.imShape

        if names is None:
            names = tagID

        if len(imShape) == 2:
            imShape = (1,) + imShape
            meta = [self.meta]
        else:
            meta = self.meta

        if not isinstance(tagID, list):
            tagID = [tagID]

        for t, tag in enumerate(tagID):

            if not names[t] in metaVals:
                metaVals[names[t]] = []

            for i in range(imShape[0]):
                metaVals[names[t]].append(meta[i][tag].value)

        return metaVals

    def rescale_intensity(self, im, outRange):

        if im.ndim == 2:
            out = rescale_intensity(image=im, out_range=outRange)
        else:
            out = np.zeros_like(im)
            for i in range(out.shape[0]):
                out[i, ...] = rescale_intensity(image=im[i, ...], out_range=outRange)

        return out

    def invert_image(self, rangeToScale=None):

        import skimage
        dtype = self.I.dtype

        if not rangeToScale is None and not isinstance(rangeToScale, list):
            if self.I.ndim > 2:
                rangeToScale = [rangeToScale] * self.imShape[0]
            else:
                rangeToScale = [rangeToScale]

        if (self.I.ndim > 2):

            for i in range(self.imShape[0]):

                tmp = self.I[i, ...].astype(np.float64)
                tmp = (tmp - tmp.min()) / tmp.max()
                tmp = skimage.util.invert(tmp)
                # tmp = histeq(rescale_intensity(tmp, out_range = rangeToScale[i]).astype(np.uint16))
                if not rangeToScale is None:
                    self.I[i, ...] = rescale_intensity(tmp, out_range=rangeToScale[i])
                else:
                    self.I[i, ...] = rescale_intensity(tmp, out_range=(0, self.I[i, ...].max()))
                self.I[i, ...] = self.I[i, ...].astype(dtype)

        else:

            tmp = self.I.astype(np.float64)
            tmp = (tmp - tmp.min) / tmp.max()
            tmp = skimage.util.invert(tmp)

            if not rangeToScale is None:
                self.I = rescale_intensity(tmp, out_range=rangeToScale[0])
            else:
                self.I = rescale_intensity(tmp, out_range=(0, self.I.max()))

    def match_histograms_scan(self, dst, debug=False):

        out = np.zeros(self.imShape, dtype=self.I.dtype)

        if (self.I.ndim > 2):

            for i in range(self.imShape[0]):

                out[i, ...] = self.match_histograms(image=self.I[i, ...].astype(np.uint16), reference=dst[i, ...])

                if (debug):
                    self.calc_hist(show=False, im=dst[i, ...], title='dst_before')
                    self.calc_hist(show=False, im=out[i, ...], title='transformed')

        else:
            out = self.match_histograms(image=self.I, reference=dst)

        self.I = out

        return out

    def hist_equalization_cv(self, dst=None, debug=False):

        out = np.zeros(self.imShape, dtype=self.I.dtype)

        if (self.I.ndim > 2):
            for i in range(self.imShape[0]):
                img = rescale_intensity(self.I[i, ...], out_range=(0, 255)).astype(np.uint8)
                imgDst = rescale_intensity(dst[i, ...], out_range=(0, 255)).astype(np.uint8)
                out_ = cv2.equalizeHist(src=img, dst=imgDst).astype(self.I.dtype)
                out_ = rescale_intensity(out_, out_range=(dst[i, ...].min(), dst[i, ...].max()))
                out[i, ...] = out_

                if (debug):
                    self.calc_hist(show=False, im=dst[i, ...], title='dst_before')
                    self.calc_hist(show=False, im=imgDst, title='dst')
                    self.calc_hist(im=img, title='transformed')

        else:
            img = rescale_intensity(self.I, out_range=(0, 255)).astype(np.uint8)
            imgDst = rescale_intensity(dst, out_range=(0, 255)).astype(np.uint8)
            out_ = cv2.equalizeHist(src=img, dst=imgDst)
            out = rescale_intensity(out_, out_range=(dst.min(), dst.max()))

        self.I = out

        return out

    def _match_cumulative_cdf(self, source, template):
        """
        Return modified source array so that the cumulative density function of
        its values matches the cumulative density function of the template.
        """
        src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                               return_inverse=True,
                                                               return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

        return interp_a_values[src_unique_indices].reshape(source.shape)

    def match_histograms(self, image, reference, multichannel=False):

        """Adjust an image so that its cumulative histogram matches that of another.

        The adjustment is applied separately for each channel.

        Parameters
        ----------
        image : ndarray
            Input image. Can be gray-scale or in color.
        reference : ndarray
            Image to match histogram of. Must have the same number of channels as
            image.
        multichannel : bool, optional
            Apply the matching separately for each channel.

        Returns
        -------
        matched : ndarray
            Transformed input image.

        Raises
        ------
        ValueError
            Thrown when the number of channels in the input image and the reference
            differ.

        References
        ----------
        .. [1] http://paulbourke.net/miscellaneous/equalisation/

        """
        shape = image.shape
        image_dtype = image.dtype

        if image.ndim != reference.ndim:
            raise ValueError('Image and reference must have the same number of channels.')

        if multichannel:
            if image.shape[-1] != reference.shape[-1]:
                raise ValueError('Number of channels in the input image and reference '
                                 'image must match!')

            matched = np.empty(image.shape, dtype=image.dtype)
            for channel in range(image.shape[-1]):
                matched_channel = self._match_cumulative_cdf(image[..., channel], reference[..., channel])
                matched[..., channel] = matched_channel
        else:
            matched = self._match_cumulative_cdf(image, reference)

        return matched

    def create_noisy_with_func(self, paramFileName, doseReduction):

        self.read_parameters_for_fun(paramFileName)
        sigmay = self.func(self.I)
        newApproxVal = self.I / doseReduction
        realStd = self.func(newApproxVal)
        qt = sigmay / realStd
        noiseSigma = sigmay ** 2 * ((doseReduction ** 2 - qt ** 2) / (doseReduction ** 2 * qt ** 2))
        noiseSigma[noiseSigma < 0] = 0
        N = np.random.normal(0, np.sqrt(noiseSigma))
        self.noisy = (self.I / doseReduction) + N

        return self.noisy, np.sqrt(noiseSigma), N
        # self.noisy = self.rescale_intensity(im = self.noisy, outRange = (self.I.min(), self.I.max()))
        # self.noisy = self.match_histograms_scan(dst = self.I)

    def funcLog(self, x, L=2 ** 14):
        return ((L - 1) / np.log(L)) * np.log(x + 1)

    def funcSqrt(self, x, L=2 ** 14):
        return np.sqrt(x * (L - 1))

    def normalize(self, x):

        return (x - x.min()) / x.max()

    def func(self, x, Type=''):

        # x = self.normalize(x)
        if self.conversionMethod == 'func':

            if self.funcName is None:

                if 'CC' in Type:
                    params = self.funcParamsCC
                elif 'MLO' in Type:
                    params = self.funcParamsMLO
                else:
                    params = self.funcParams
                out = [param * x ** (len(params) - 1 - i) for i, param in enumerate(params)]
                return np.sum(out, axis=0)

            else:

                out = getattr(self, self.funcName)(x)
                return out

        else:
            if ('CC' in Type):
                lookUpTable = self.lookUpTableCC
            else:
                lookUpTable = self.lookUpTableMLO
            return self.process_pixels_with_lookuptable(x, lookUpTable)

    def process_pixels_with_lookuptable(self, x, dict):

        xflatten = x.flatten()
        y = np.zeros(xflatten.shape)

        for i in range(xflatten.shape[0]):

            if (xflatten[i] in dict):
                y[i] = dict[xflatten[i]]
            else:
                y[i] = 0
                # j = i
                # while not xflatten[j] in dict:
                #     j += 1
                # y[i] = dict[xflatten[j]]

        y = y.reshape(x.shape)

        return y

    def merge_scans(self, basePathScans, outDir, dirFilter=None):

        k = 0

        for d in os.listdir(basePathScans):

            if dirFilter is not None and dirFilter not in d:
                continue

            I, meta = self.dicom.readDicomSeriesWithMeta(os.path.join(basePathScans, d))
            if not isinstance(meta, list):
                meta = [meta]
            fileNames = ['{}.dcm'.format(str(x)) for x in range(k, k + len(meta))]
            k += len(meta)
            self.dicom.writeDicomSeriesUsingMeta(I,
                                                 meta,
                                                 outDir,
                                                 description=meta[0].SeriesDescription,
                                                 fileNames=fileNames)

    def compute_quality_index(self, index, **args):

        if index == 'cnn' and args.get('cnnName', None) is None:
            raise RuntimeError('Cannot run index quality with cnn without cnn name')

        inputArgs = inspect.getfullargspec(getattr(self, f'compute_{index}_')).args
        args_ = {key: args[key] for key in inputArgs if key in args}
        result = getattr(self, f'compute_{index}_')(**args_)

        return result

    def import_model(self, name, vggmodel=None):

        if not vggmodel is None:
            model = getattr(self.nets, name, None)(vggmodel)
        else:
            model = getattr(self.nets, name, None)
        if model is None:
            raise RuntimeError('no such model {}'.format(name))
        return model

    def compute_index_cnn(self, cnnName):

        import torch
        import importlib
        self.nets = importlib.import_module('pytorch.pytorch_nets')

        if 'CPU' in cnnName:
            cnnName = cnnName.replace('CPU', '')
            useCPU = True
        else:
            useCPU = False

        if 'vgg' in cnnName or 'hyper' in cnnName:
            from torchvision.models import vgg
            vgg_model = vgg.vgg16(pretrained=True)
            if not useCPU and torch.cuda.is_available():
                vgg_model.cuda()
            model = self.import_model(cnnName, vgg_model)
        else:
            model = self.import_model(cnnName)

        model.eval()
        if not useCPU:
            model.cuda()
        O = []

        for sliceNum in range(self.imShape[0]):
            I = self.I[sliceNum, :, :][None, None, :, :]
            reco = self.reconstructed[sliceNum, :, :][None, None, :, :]
            if not useCPU:
                I = torch.autograd.Variable(torch.from_numpy(I.astype('float32')).cuda())
                reco = torch.autograd.Variable(torch.from_numpy(reco.astype('float32')).cuda())
            else:
                I = torch.autograd.Variable(torch.from_numpy(I.astype('float32')))
                reco = torch.autograd.Variable(torch.from_numpy(reco.astype('float32')))

            with torch.no_grad():
                I = model(I)
                reco = model(reco)

            L = torch.nn.MSELoss()(I, reco)
            if useCPU:
                L = L.data.numpy()
            else:
                L = L.cpu().data.numpy()
            O.append(L)

        return np.mean(O)

    def close(self):
        plt.close()

    def add_ROI(self, pos):
        self.ROIS.append(pos)

    def draw_roi(self, im=None, cmap=None, TH=None):

        if im.ndim == 2:
            im = im[None, ...]

        roi = np.zeros_like(im)
        sliceNum = 0

        while sliceNum < im.shape[0]:

            self.get_pos(im[sliceNum, ...], cmap, selector='lasso')
            mask = Image.new('L', im[sliceNum, ...].shape, 0)
            ImageDraw.Draw(mask).polygon(self.pos, outline=1, fill=True)
            mask = np.array(mask)
            if not TH is None:
                mask[im[sliceNum, ...] < TH] = 0
                mask = largest_connected_component(mask)
                mask = binary_opening(mask, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
                mask = fill_foles(mask)
            if np.sum(mask) > 1:
                roi[sliceNum, ...] = mask
                sliceNum += 1

        return roi

    def extract_patches_from_poses(self, sliceNum=0, poses=None, pixelsX=None, pixelsY=None):

        poses = getattr(self, 'poses', poses)

        twoD = False
        patches = []
        pixelsX = np.arange(0, 1) if pixelsX is None else pixelsX
        pixelsY = np.arange(0, 1) if pixelsY is None else pixelsY

        if self.I.ndim == 2:
            self.I = self.I[None, :, :]
            twoD = True

        for pos in poses:

            pos = [int(x) for x in pos]

            for pixelX in pixelsX:
                for pixelY in pixelsY:
                    posX = pos[0] + pixelX
                    posY = pos[1] + pixelY

                    patches.append(self.I[sliceNum:sliceNum + 1, posY:posY + pos[3], posX:posX + pos[2]])

        if twoD:
            self.I = self.I[0, ...]

        patches = np.concatenate(patches, axis=0)

        return patches

    def get_pos_3D(self, z, depth, **kwargs):

        im = kwargs.get('im', None)
        kwargs['im'] = self.I[z, ...] if im is None else im
        self.get_pos(**kwargs)
        self.pos = [self.pos[0], self.pos[1], z, self.pos[2], self.pos[3], depth]

        for i, p in enumerate(self.poses):
            self.poses[i] = [p[0], p[1], z, p[2], p[3], depth]

    def get_pos(self, im=None, cmap=None, initialPos=None, multipleSelectKey=None, selector='rectangle'):

        self.selector = selector
        self.posed = False
        self.poses = []

        if (im is None):
            im = self.I

        if (cmap is None):
            cmap = self.cmap

        def ellipse_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            self.pos = get_ellipse_inds(c=(x2 - x1) / 2 + x1, r=(y2 - y1) / 2 + y1, r_radius=(y2 - y1) / 2,
                                        c_radius=(x2 - x1) / 2)
            self.Ellipse = Ellipse(xy=((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1), width=x2 - x1, height=y2 - y1)
            self.posed = True

        def lasso_select_callback(verts):
            self.pos = verts

        def line_select_callback(eclick, erelease):
            # 'eclick and erelease are the press and relea      se events'
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            pos = np.array([x1, y1, x2 - x1, y2 - y1])
            self.pos = [np.int(x) for x in pos]

        def toggle_selector(event):
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                toggle_selector.RS.set_active(True)

        def press(event):
            print('pressed', event.key)
            if event.key == 'enter' or event.key == 'space':
                plt.close()
            if (not hasattr(self, 'poses')):
                self.poses = []
            if (event.key == multipleSelectKey):
                print('save pos')
                self.poses.append(self.pos)

        fig, current_ax = plt.subplots()
        current_ax.imshow(im, cmap=cmap, clim=self.clim)
        # drawtype is 'box' or 'line' or 'none'
        if self.selector == 'rectangle':
            toggle_selector.RS = RectangleSelector(current_ax, line_select_callback, drawtype='box', useblit=False,
                                                   button=[1, 3], spancoords='pixels', interactive=True)
        elif self.selector == 'lasso':
            toggle_selector.RS = LassoSelector(current_ax, lasso_select_callback, useblit=False, button=[1, 3], )
        elif self.selector == 'ellipse':
            toggle_selector.RS = EllipseSelector(current_ax, ellipse_select_callback, useblit=False, button=[1, 3],
                                                 drawtype='box', interactive=True, spancoords='pixels')

        plt.connect('key_press_event', toggle_selector)
        plt.connect('key_press_event', press)

        if (not initialPos is None):
            toggle_selector.RS.to_draw.set_visible(True)
            if selector == 'rectangle':
                toggle_selector.RS.extents = (0, initialPos[0], 0, initialPos[1])
            if selector == 'ellipse':
                fields = ['_center', 'width', 'height']
                toggle_selector.RS.extents = (initialPos._center[0], 0, initialPos._center[1], 0)
                for field in fields:
                    setattr(toggle_selector.RS.to_draw, field, getattr(initialPos, field))

        plt.show()

        return self.pos

    def imtool(self, im=None, sliceNum=None, Range=None):

        sliceNum = 0 if sliceNum is None else sliceNum
        if im.ndim == 3:
            im = self.I[sliceNum, ...] if im is None else im[sliceNum, ...]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.25, bottom=0.25)

        if Range is None:
            min0 = im.min()
            max0 = im.max()
        else:
            min0 = Range[0]
            max0 = Range[1]

        im1 = ax.imshow(im, cmap='gray')
        fig.colorbar(im1)

        axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03])

        if Range is None:
            smin = Slider(axmin, 'Min', 0, im.max(), valinit=min0)
            smax = Slider(axmax, 'Max', 0, im.max(), valinit=max0)
        else:
            smin = Slider(axmin, 'Min', 0, Range[0], valinit=Range[0])
            smax = Slider(axmax, 'Max', 0, Range[1], valinit=Range[1])

        def update(val):
            im1.set_clim([smin.val, smax.val])
            fig.canvas.draw()

        smin.on_changed(update)
        smax.on_changed(update)

        plt.show()
