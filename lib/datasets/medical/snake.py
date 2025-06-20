import torch.utils.data as data
from lib.utils.snake import snake_kins_utils, snake_config
import numpy as np
import math
from lib.utils import data_utils
import os
from lib.config import cfg
from dicom.my_dicom import myDicom
from lib.image.my_image import myImage
import cv2

dicom = myDicom()
image = myImage()

FIXED_INPUT_SIZE = (512, 512)  # (height, width)

class Dataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, sub_folder,split='train'):
        super(Dataset, self).__init__()

        # self.data_root = data_root
        # self.split = split
        #
        # self.coco = COCO(ann_file)
        # self.anns = np.array(self.coco.getImgIds())
        # self.anns = self.anns[:500] if split == 'mini' else self.anns
        # self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.split = split
        self.sub_folder = sub_folder
        self.slice_list = []

        with open('/media/cilab/DATA/Hila/Projects/Fatty_Liver/fatty_liver/datasets/valid_mri_scans_3T.txt', 'r') as file:
            lines = file.readlines()
        cases = [line.split() for line in lines]
        cases_flat = [item for sublist in cases for item in sublist]

        for case in cases_flat:
            volume_id = case
            img_path = os.path.join(self.image_dir, case, self.sub_folder)
            mask_path = os.path.join(self.mask_dir, case)

            img = image.read_im(img_path)
            mask = image.read_im(mask_path)

            num_slices = img.shape[0]

            for i in range(num_slices):
                self.slice_list.append((volume_id,img_path, mask_path, i))

    def __len__(self):
        return len(self.slice_list)


    def get_slice(self, scan_vol, mask_vol, idx):
            return scan_vol[idx, :, :], mask_vol[idx, :, :]

    def read_slice_data(self, scan_path, mask_path, idx):
        img = image.read_im(scan_path)
        mask = image.read_im(mask_path)
        img_slice, mask_slice = self.get_slice(img, mask, idx)

        # Normalize image
        img_slice = np.clip(img_slice, np.percentile(img_slice, 1), np.percentile(img_slice, 99))
        img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-5)) * 255.0
        img = img_slice.astype(np.uint8)
        mask = (mask_slice > 0).astype(np.uint8) * 255

        # Resize image and mask to fixed size
        img = cv2.resize(img, FIXED_INPUT_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, FIXED_INPUT_SIZE[::-1], interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        instance_polys = []
        for contour in contours:
            if len(contour) >= 4:
                poly = contour.squeeze(1).astype(np.float32)
                instance_polys.append([poly])

        cls_ids = [0 for _ in instance_polys]  # single class
        return img, instance_polys, cls_ids

    # def process_info(self, index):
    #     # ann_ids = self.coco.getAnnIds(imgIds=img_id)
    #     # anno = self.coco.loadAnns(ann_ids)
    #     # path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
    #     # return anno, path, img_id
    #
    #     img_path, mask_path = self.slice_paths[index]
    #     return mask_path, img_path, index
    #
    # def read_original_data(self, mask_path, img_path):
    #     img = self.read_dicom_image(img_path)
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #
    #     # Load mask
    #     mask = self.read_dicom_image(mask_path)
    #     mask = (mask > 127).astype(np.uint8) * 255
    #
    #     # Extract contours
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     instance_polys = []
    #     for contour in contours:
    #         if len(contour) >= 4:
    #             poly = contour.squeeze(1).astype(np.float32)
    #             instance_polys.append([poly])
    #
    #     cls_ids = [0 for _ in instance_polys]  # Single class
    #
    #     return img_rgb, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = instance

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_kins_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys):
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            polys = snake_kins_utils.filter_tiny_polys(instance)
            polys = snake_kins_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            polys = [poly for poly in polys if len(poly) >= 4]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_kins_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_kins_utils.get_init(box)
        img_init_poly = snake_kins_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_kins_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_kins_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_kins_utils.get_octagon(extreme_point)
        img_init_poly = snake_kins_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_kins_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_kins_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_kins_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)

    def __getitem__(self, index):
        volume_id, scan_path, mask_path, slice_idx = self.slice_list[index]
        img, instance_polys, cls_ids = self.read_slice_data(scan_path, mask_path, slice_idx)

        height, width = img.shape[:2]

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_kins_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )

        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys)
        extreme_points = self.get_extreme_points(instance_polys)

        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh, ct_cls, ct_ind = [], [], []

        i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys = [], [], [], []
        i_it_pys, c_it_pys, i_gt_pys, c_gt_pys = [], [], [], []

        for i in range(len(instance_polys)):
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]
            cls_id = cls_ids[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h,
                                  output_w)
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        ret = {
            'inp': inp,
            'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind,
            'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys,
            'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys,
            'i_it_py': i_it_pys, 'c_it_py': c_it_pys,
            'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys,
            'meta': {
                'center': center,
                'scale': scale,
                'img_id': slice_idx,
                'volume_id': volume_id,
                'slice_idx': slice_idx,
                'ct_num': len(ct_ind)
            }
        }

        return ret




