import sys
import os
# from image.my_image import myImage
import numpy as np
# import cv2
import scipy.ndimage
from scipy.interpolate import splprep, splev
import skimage.morphology
from skimage import measure
import matplotlib.pyplot as plt


# def read_seg(path):
#     seg_img = myImage()
#     seg_img.read_im(path)  # path to binary seg of the prostate
#     return seg_img


# not in use:
def b_spline_to_bezier_series(tck, per = False):
  """Convert a parametric b-spline into a sequence of Bezier curves of the same degree.

  Inputs:
    tck : (t,c,k) tuple of b-spline knots, coefficients, and degree returned by splprep.
    per : if tck was created as a periodic spline, per *must* be true, else per *must* be false.

  Output:
    A list of Bezier curves of degree k that is equivalent to the input spline.
    Each Bezier curve is an array of shape (k+1,d) where d is the dimension of the
    space; thus the curve includes the starting point, the k-1 internal control
    points, and the endpoint, where each point is of d dimensions.
  """
  from scipy.interpolate.fitpack import insert
  from numpy import asarray, unique, split, sum
  t,c,k = tck
  t = asarray(t)
  try:
    c[0][0]
  except:
    # I can't figure out a simple way to convert nonparametric splines to
    # parametric splines. Oh well.
    raise TypeError("Only parametric b-splines are supported.")
  new_tck = tck
  if per:
    # ignore the leading and trailing k knots that exist to enforce periodicity
    knots_to_consider = unique(t[k:-k])
  else:
    # the first and last k+1 knots are identical in the non-periodic case, so
    # no need to consider them when increasing the knot multiplicities below
    knots_to_consider = unique(t[k+1:-k-1])
  # For each unique knot, bring it's multiplicity up to the next multiple of k+1
  # This removes all continuity constraints between each of the original knots,
  # creating a set of independent Bezier curves.
  desired_multiplicity = k+1
  for x in knots_to_consider:
    current_multiplicity = sum(t == x)
    remainder = current_multiplicity%desired_multiplicity
    if remainder != 0:
      # add enough knots to bring the current multiplicity up to the desired multiplicity
      number_to_insert = desired_multiplicity - remainder
      new_tck = insert(x, new_tck, number_to_insert, per)
  tt,cc,kk = new_tck
  # strip off the last k+1 knots, as they are redundant after knot insertion
  bezier_points = np.transpose(cc)[:-desired_multiplicity]
  if per:
    # again, ignore the leading and trailing k knots
    bezier_points = bezier_points[k:-k]
  # group the points into the desired bezier curves
  return split(bezier_points, len(bezier_points) / desired_multiplicity, axis = 0)


# not in use:
def get_curve_with_bezier_func(img):

    (d, h, w) = img.shape
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    contour = np.zeros(img.shape)
    coord = list()  # list of arrays, for each slice an array with XY for the contour
    coord_len = list()
    for z in range(0, d):
        # try with openCV
        im2, contours, hierarchy = cv2.findContours(img[z], 1, 2)  # gets points in order
        if len(contours) > 0:
            # TODO: need to decide what type of curve need to be insert to header
            pts = contours[0][:,0,:]
            # coord.append(bytes(pts.flatten('C').astype(np.uint8)))
            coord.append(bytes(pts.flatten('C').astype(np.single)))
            coord_len.append(len(pts))
            # TODO: need to be sure if need BEZIER type for insert to header, a function that do it..

            # tck, u = splprep(pts.T, u=None, s=3.0, per=1, k=5)
            # bezier = b_spline_to_bezier_series(tck, per=True)
            # pass

            # u_new = np.linspace(u.min(), u.max(), 1000)
            # x_new, y_new = splev(u_new, tck, der=0)
            # coord.append(bytes(np.array([x_new, y_new]).T.flatten('C').astype(np.uint8)))
            # coord_len.append(len(x_new))
        else:
            coord.append(contours)
            coord_len.append(0)
        # getting the contour by morphology, points not in order
        # down the object size by one the reduce regular-down

        # erosion = cv2.erode(img[z], kernel, iterations=1)
        # contour[z] = img[z] - erosion
        # # TODO: need to check if the coord is [x,y] or [y,x] after add it to Dicom
        # #dummy_curve = b'\xb54\x18C \x1e\xedB\xd8\xd7\xeeB\xe8f\nC\x97\x96\xddB\x16\x94;C#\xa3\x02C\xad*jC\x19\x99(C\xad*jC\x14\x94;C\xc5CcCC\xc1lC$\xa3ZC\xe4auCo\xeeEC\xa2 dC\x0b\x8a\tCG\xc6YC\x8e\x0c\x00C\xb54DC\xd9\xd7\xeeB\xb54\x18Cfd\xebB'
        # # coord.append(np.transpose(np.nonzero(contour[z])).astype(np.uint8).tobytes('C'))
        # #c = (np.array(np.transpose(np.nonzero(contour[z])).flatten('C').astype(np.uint8),dtype='<b'))
        #
        # coord.append(bytes(np.array(np.transpose(np.nonzero(contour[z])).flatten('C').astype(np.uint8),dtype='<b')))
        # coord_len.append(np.shape(np.transpose(np.nonzero(contour[z])))[0])
    # coord = map(lambda s: s.strip(), coord)
    # coord_strip = list(map(str.strip, coord))
    return coord, coord_len


# in use:
def get_curve(img):
    '''
    takes binary volume that represent an segmentation of an 3D object
    and find for every slice the contour of the object
    it uses cv2.findContours so the image must be uint8 or uint32
    '''
    (d, h, w) = img.shape
    coord = list()  # list of arrays, for each slice an array with XY for the contour
    coord_len = list()
    for z in range(0, d):

        coord.append([])
        coord_len.append([])

        # morphology operator:
        if np.unique(img[z]).shape[0] == 1:
            coord[z].append(None)
            coord_len[z].append(0)
            continue

        contours = measure.find_contours(img[z], 0.5)

        for pts in contours:

            pts = np.vstack([pts, pts[0]])  # add the first point to be the last for close contour
            pts = pts[:, ::-1]
            coord[z].append(bytes(pts.flatten('C').astype(np.single)))  # adds the curve in type of single
            coord_len[z].append(len(pts))

    return coord, coord_len


# # for testing:
# if __name__ == '__main__':
#     seg_path = os.path.dirname(__file__) + r'/Output/1'
#     T2_path = os.path.dirname(__file__) + r'/Data/RADASHKOVSKY_ARKADDI'
#     seg_im = read_seg(seg_path)
#     coord, coord_len = get_curve(seg_im.I.astype(np.uint8))
#     pass
