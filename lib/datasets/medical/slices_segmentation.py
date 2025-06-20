import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from image.my_image import myImage
from dicom.my_dicom import myDicom
from lib.utils.snake import snake_config

my_dicom = myDicom()
my_image = myImage()

class Dataset(Dataset):
    def __init__(self, scan_dir, init_mask_dir, gt_mask_dir, sub_folder,is_train=True, transforms=None):
        self.scan_dir = scan_dir
        self.init_mask_dir = init_mask_dir
        self.gt_mask_dir = gt_mask_dir
        self.transforms = transforms
        self.is_train = is_train
        self.sub_folder = sub_folder

        self.pairs = self._get_case_pairs()
        self.slice_index = []
        self._build_slice_index()

    def _get_case_pairs(self):
        if self.is_train == True:
            gt_cases = sorted([d for d in os.listdir(self.gt_mask_dir) if os.path.isdir(os.path.join(self.gt_mask_dir, d))])
            scan_cases = sorted([d for d in os.listdir(self.gt_mask_dir) if os.path.isdir(os.path.join(self.scan_dir, d,self.sub_folder))])
            init_cases = sorted([d for d in os.listdir(self.gt_mask_dir) if os.path.isdir(os.path.join(self.init_mask_dir, d))])
            common_cases = sorted(set(scan_cases) & set(init_cases) & set(gt_cases))

        else:
            init_cases = sorted([d for d in os.listdir(os.path.join(self.init_mask_dir)) if os.path.isdir(os.path.join(self.init_mask_dir, d))])
            scan_cases = sorted([d for d in os.listdir(os.path.join(self.init_mask_dir)) if os.path.isdir(os.path.join(self.scan_dir,d,self.sub_folder))])
            common_cases = sorted(set(scan_cases) & set(init_cases))

        pairs = []
        for case in common_cases:
            scan_path = os.path.join(self.scan_dir, case,self.sub_folder)
            init_mask_path = os.path.join(self.init_mask_dir, case)
            if self.gt_mask_dir is not None:
                gt_mask_path = os.path.join(self.gt_mask_dir, case)
                pairs.append((scan_path, init_mask_path, gt_mask_path))
            else:
                pairs.append((scan_path, init_mask_path,None))

        return pairs

    def _build_slice_index(self):
        for i, (scan_path, _, _) in enumerate(self.pairs):
            scan = self._load_dicom_series(scan_path)
            num_slices = scan.shape[0]
            for j in range(num_slices):
                self.slice_index.append((i, j))

    def _load_dicom_series(self, folder_path):
        return my_image.read_im(folder_path)  # shape: [slices, H, W]

    def _extract_contour_from_mask(self, mask_slice):
        mask_bin = (mask_slice > 0).astype(np.uint8)
        _,contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # return contours
        contour_mask = np.zeros_like(mask_bin)
        cv2.drawContours(contour_mask, contours, -1, color=1, thickness=1)
        return contour_mask, len(contours)

    def mask_to_polygon(self, mask_slice):
        _,contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)  # dummy box

        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            polygon = np.array(box, dtype=np.float32)
        else:
            polygon = approx[:, 0, :].astype(np.float32)

        # Ensure exactly 4 points
        if polygon.shape[0] > 4:
            polygon = polygon[:4]
        elif polygon.shape[0] < 4:
            pad = 4 - polygon.shape[0]
            polygon = np.concatenate([polygon, np.tile(polygon[-1:], (pad, 1))], axis=0)

        return polygon

    # def get_bounding_box_from_contour(self, contour):

    def normalize_polygon(self, polygon, center, size):
        norm_poly = (polygon - center) / size  # (x,y) normalized by (w,h)
        return norm_poly.astype(np.float32)

    def mask_to_contour(self, mask_slice):
        # Extract all contour points (no polygon approximation)
        _, contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            # return a dummy small polygon
            return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

        cnt = max(contours, key=cv2.contourArea)
        polygon = cnt[:, 0, :].astype(np.float32)  # Nx2 polygon with all contour points
        return polygon

    import cv2
    import numpy as np

    def generate_init_polygon(self,mask, num_points):
        """
        mask: binary numpy array (H, W), 0 background, 1 foreground
        Returns: (num_points, 2) polygon points
        """
        # Step 1: Find contours
        _,contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # if len(contours) == 0:
        #     return None
        #     # raise ValueError("No contours found in mask")

        # Choose largest contour if multiple
        contour = max(contours, key=cv2.contourArea).squeeze(1)  # shape: (N, 2)

        # Step 2: Compute arc length and resample uniformly
        arc_len = cv2.arcLength(contour, closed=True)
        epsilon = 0.001 * arc_len
        approx = cv2.approxPolyDP(contour, epsilon, closed=True).squeeze(1)

        # Interpolate to get fixed number of points
        contour = np.vstack([contour, contour[0]])  # Close the contour
        dists = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
        dists = np.insert(dists, 0, 0)
        uniform_dists = np.linspace(0, dists[-1], num_points)
        interp_pts = np.empty((num_points, 2), dtype=np.float32)
        for i in range(2):
            interp_pts[:, i] = np.interp(uniform_dists, dists, contour[:, i])

        return interp_pts  # shape: (num_points, 2)

    def normalize_init_polygon(self,polygon, img_width, img_height):
        norm = np.zeros_like(polygon, dtype=np.float32)
        norm[:, 0] = polygon[:, 0] / img_width
        norm[:, 1] = polygon[:, 1] / img_height
        return norm

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        pair_idx, slice_idx = self.slice_index[idx]
        scan_path, init_mask_path, gt_mask_path = self.pairs[pair_idx]

        # Load full 3D volumes
        scan_vol = self._load_dicom_series(scan_path)
        init_mask_vol = self._load_dicom_series(init_mask_path)
        gt_mask_vol = self._load_dicom_series(gt_mask_path)

        # Get 2D slices
        scan_slice = scan_vol[slice_idx, :, :].astype(np.float32)
        init_mask_slice = init_mask_vol[slice_idx, :, :].astype(np.uint8)
        gt_mask_slice = gt_mask_vol[slice_idx, :, :].astype(np.uint8)

        # Extract initial contour from initial mask
        init_contour,number_of_contours = self._extract_contour_from_mask(init_mask_slice)
        H, W = scan_slice.shape
        image_tensor = torch.from_numpy(scan_slice).unsqueeze(0)

        # Initial mask polygon
        # i_it_4py = self.mask_to_polygon(init_mask_slice)  # initial polygon in image space

        init_poly_num = snake_config.init_poly_num
        poly_num = snake_config.poly_num

        # if self.is_train:
        #     poly_num = snake_config.init_poly_num
        # else:
        #     poly_num = snake_config.poly_num

        if number_of_contours > 0:
            i_it_4py = self.generate_init_polygon(init_mask_slice,init_poly_num)
            # i_it_py = self.mask_to_contour(init_mask_slice)   # initial contour points
            i_it_py = self.generate_init_polygon(init_mask_slice, poly_num)
        else:
            i_it_4py = np.empty((init_poly_num, 2))
            i_it_py = np.empty((poly_num, 2))

        # Get bounding box from contour
        ys, xs = np.where(init_mask_slice > 0)
        if len(xs) == 0 or len(ys) == 0:
            ct_x, ct_y = W // 2, H // 2
            w, h = 1, 1
        else:
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            ct_x = (x1 + x2) / 2
            ct_y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

        # Generate heatmap
        ct_hm = np.zeros((1, H, W), dtype=np.float32)
        radius = max(2, int(min(w, h) / 2))
        ct_x_int, ct_y_int = int(ct_x), int(ct_y)
        cv2.circle(ct_hm[0], (ct_x_int, ct_y_int), radius, 1, -1)

        ct_ind = ct_y_int * W + ct_x_int  # index in flattened feature map
        wh = np.array([[w, h]], dtype=np.float32)
        ct_cls = np.array([0], dtype=np.int64)  # class index (0 if single-class)
        center = np.array([ct_x, ct_y])
        size = np.array([w, h]) + 1e-6  # avoid divide-by-zero

        if number_of_contours>0:
            # c_it_4py = self.normalize_init_polygon(i_it_4py, W, H)
            c_it_4py = self.normalize_init_polygon(i_it_4py, W, H)
            c_it_py = self.normalize_init_polygon(i_it_py, W, H)
            # c_it_py = self.normalize_polygon(i_it_py, center, size)
        else:
            c_it_4py = np.empty((init_poly_num, 2))
            c_it_py = np.empty((poly_num, 2))

        # if self.is_train:
        gt_contour, number_of_contours= self._extract_contour_from_mask(gt_mask_slice)
        if number_of_contours > 0:
            # i_gt_4py = self.generate_init_polygon(gt_mask_slice,poly_num)
            i_gt_4py = self.mask_to_polygon(gt_mask_slice)  # ground-truth polygon in image space
            i_gt_py = self.generate_init_polygon(gt_mask_slice,poly_num)

            # i_gt_py = self.mask_to_polygon(gt_mask_slice)  # ground-truth polygon in image space
        else:
            i_gt_4py = np.empty((4, 2))
            i_gt_py = np.empty((poly_num, 2))


        # Get bounding box from contour
        ys, xs = np.where(gt_mask_slice > 0)
        if len(xs) == 0 or len(ys) == 0:
            ct_x, ct_y = W // 2, H // 2
            w, h = 1, 1
        else:
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            ct_x = (x1 + x2) / 2
            ct_y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

        # Generate heatmap
        ct_hm = np.zeros((1, H, W), dtype=np.float32)
        radius = max(2, int(min(w, h) / 2))
        ct_x_int, ct_y_int = int(ct_x), int(ct_y)
        cv2.circle(ct_hm[0], (ct_x_int, ct_y_int), radius, 1, -1)

        ct_ind = ct_y_int * W + ct_x_int  # index in flattened feature map
        wh = np.array([[w, h]], dtype=np.float32)
        ct_cls = np.array([0], dtype=np.int64)  # class index (0 if single-class)

        if number_of_contours > 0:
            c_gt_4py = self.normalize_init_polygon(i_gt_4py, W, H)
            # c_gt_4py = self.normalize_polygon(i_gt_4py, center, size)
            c_gt_py = self.normalize_polygon(i_gt_py, center, size)
        else:
            c_gt_4py = np.empty((4, 2))
            c_gt_py = np.empty((poly_num, 2))

        # Normalize to canonical space (centered on box)
        center = np.array([ct_x, ct_y])
        size = np.array([w, h]) + 1e-6  # avoid divide-by-zero

        meta = {
            'ct_num': 1,
        }

        sample = {
            'inp': image_tensor,  # input image
            'init_contour': init_contour,  # initial contour
            'gt_contour': gt_contour,  # ground truth
            'ct_hm': torch.from_numpy(ct_hm),  # center heatmap
            'wh': torch.from_numpy(wh),  # width-height at center
            'ct_cls': torch.from_numpy(ct_cls),  # center class
            'ct_ind': torch.tensor([ct_ind]),  # center index
            'meta': meta,  # meta info
            'i_it_4py': torch.from_numpy(i_it_4py),
            'c_it_4py': torch.from_numpy(c_it_4py),
            'i_gt_4py': torch.from_numpy(i_gt_4py),
            'c_gt_4py': torch.from_numpy(c_gt_4py),
            'i_it_py': torch.from_numpy(i_it_py),
            'c_it_py': torch.from_numpy(c_it_py),
            'i_gt_py': torch.from_numpy(i_gt_py),
            'c_gt_py': torch.from_numpy(c_gt_py)
        }
        # else:
        #     sample = {
        #         'inp': image_tensor,
        #         'init_contour': init_contour,
        #         'meta': {'ct_num': 0},
        #         'i_it_4py': torch.from_numpy(i_it_4py),
        #         'c_it_4py': torch.from_numpy(c_it_4py),
        #         'i_it_py': torch.from_numpy(i_it_py),
        #         'c_it_py': torch.from_numpy(c_it_py),
        #     }

        if self.transforms:
            sample = self.transforms(sample)

        return sample





















        # import matplotlib.pyplot as plt
        # plt.imshow(init_contour);plt.show()
        # sample = {
        #     'image': scan_slice,
        #     'init_contour': init_contour,
        # }
        #
        # if self.is_train:
        #     gt_contour = self._extract_contour_from_mask(gt_mask_slice)
        #     sample['gt_contour'] = gt_contour
        #
        # if self.transforms:
        #     sample = self.transforms(sample)
        #
        # return sample
