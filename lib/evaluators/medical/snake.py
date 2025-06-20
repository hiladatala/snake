import os
import cv2
import json
import numpy as np
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils

import os
import numpy as np
from medpy.metric.binary import dc  # Dice coefficient


# from medpy.metric.binary import hd  # for Hausdorff if needed

class Evaluator:
    def __init__(self, result_dir):
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

        self.slice_metrics = []  # list of dicts: {case_id, slice_id, dice}

    def evaluate(self, pred_mask, gt_mask, case_id, slice_id):
        """
        pred_mask, gt_mask: 2D numpy arrays (binary masks)
        case_id: patient/scan ID (e.g. 'CT_0001')
        slice_id: integer or slice index
        """
        assert pred_mask.shape == gt_mask.shape, f"Shape mismatch: pred {pred_mask.shape}, gt {gt_mask.shape}"

        # Binary masks
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        # Handle empty masks
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            dice = 1.0  # perfect match if both are empty
        else:
            dice = dc(pred_mask, gt_mask)

        self.slice_metrics.append({
            'case_id': case_id,
            'slice_id': slice_id,
            'dice': dice
        })

    def summarize(self):
        # Compute mean Dice per volume and overall
        from collections import defaultdict

        volume_scores = defaultdict(list)
        for metric in self.slice_metrics:
            volume_scores[metric['case_id']].append(metric['dice'])

        mean_dice_per_volume = {k: np.mean(v) for k, v in volume_scores.items()}
        overall_mean_dice = np.mean([m['dice'] for m in self.slice_metrics])

        print(f"Overall Mean Dice: {overall_mean_dice:.4f}")
        print("Per-volume Dice:")
        for k, v in mean_dice_per_volume.items():
            print(f"  {k}: {v:.4f}")

        return {
            'mean_dice': overall_mean_dice,
            'per_volume': mean_dice_per_volume,
            'all_metrics': self.slice_metrics
        }


# class Evaluator:
#     def __init__(self, result_dir):
#         self.results = []
#         self.img_ids = []
#         self.aps = []
#
#         self.result_dir = result_dir
#         os.system('mkdir -p {}'.format(self.result_dir))
#
#         args = DatasetCatalog.get(cfg.test.dataset)
#         self.ann_file = args['ann_file']
#         self.data_root = args['data_root']
#         self.coco = coco.COCO(self.ann_file)
#
#         self.json_category_id_to_contiguous_id = {
#             v: i for i, v in enumerate(self.coco.getCatIds())
#         }
#         self.contiguous_category_id_to_json_id = {
#             v: k for k, v in self.json_category_id_to_contiguous_id.items()
#         }
#
#     def evaluate(self, output, batch):
#         detection = output['detection']
#         score = detection[:, 4].detach().cpu().numpy()
#         label = detection[:, 5].detach().cpu().numpy().astype(int)
#         py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio
#
#         if len(py) == 0:
#             return
#
#         img_id = int(batch['meta']['img_id'][0])
#         center = batch['meta']['center'][0].detach().cpu().numpy()
#         scale = batch['meta']['scale'][0].detach().cpu().numpy()
#
#         h, w = batch['inp'].size(2), batch['inp'].size(3)
#         trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
#         img = self.coco.loadImgs(img_id)[0]
#         ori_h, ori_w = img['height'], img['width']
#         py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
#         rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)
#
#         coco_dets = []
#         for i in range(len(rles)):
#             detection = {
#                 'image_id': img_id,
#                 'category_id': self.contiguous_category_id_to_json_id[label[i]],
#                 'segmentation': rles[i],
#                 'score': float('{:.2f}'.format(score[i]))
#             }
#             coco_dets.append(detection)
#
#         self.results.extend(coco_dets)
#         self.img_ids.append(img_id)
#
#     def summarize(self):
#         json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
#         coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
#         coco_eval = COCOeval(self.coco, coco_dets, 'segm')
#         coco_eval.params.imgIds = self.img_ids
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
#         self.results = []
#         self.img_ids = []
#         self.aps.append(coco_eval.stats[0])
#         return {'ap': coco_eval.stats[0]}

#
# class DetectionEvaluator:
#     def __init__(self, result_dir):
#         self.results = []
#         self.img_ids = []
#         self.aps = []
#
#         self.result_dir = result_dir
#         os.system('mkdir -p {}'.format(self.result_dir))
#
#         args = DatasetCatalog.get(cfg.test.dataset)
#         self.ann_file = args['ann_file']
#         self.data_root = args['data_root']
#         self.coco = coco.COCO(self.ann_file)
#
#         self.json_category_id_to_contiguous_id = {
#             v: i for i, v in enumerate(self.coco.getCatIds())
#         }
#         self.contiguous_category_id_to_json_id = {
#             v: k for k, v in self.json_category_id_to_contiguous_id.items()
#         }
#
#     def evaluate(self, output, batch):
#         detection = output['detection']
#         detection = detection[0] if detection.dim() == 3 else detection
#         box = detection[:, :4].detach().cpu().numpy() * snake_config.down_ratio
#         score = detection[:, 4].detach().cpu().numpy()
#         label = detection[:, 5].detach().cpu().numpy().astype(int)
#
#         img_id = int(batch['meta']['img_id'][0])
#         center = batch['meta']['center'][0].detach().cpu().numpy()
#         scale = batch['meta']['scale'][0].detach().cpu().numpy()
#
#         if len(box) == 0:
#             return
#
#         h, w = batch['inp'].size(2), batch['inp'].size(3)
#         trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
#         img = self.coco.loadImgs(img_id)[0]
#         ori_h, ori_w = img['height'], img['width']
#
#         coco_dets = []
#         for i in range(len(label)):
#             box_ = data_utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
#             box_[2] -= box_[0]
#             box_[3] -= box_[1]
#             box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
#             detection = {
#                 'image_id': img_id,
#                 'category_id': self.contiguous_category_id_to_json_id[label[i]],
#                 'bbox': box_,
#                 'score': float('{:.2f}'.format(score[i]))
#             }
#             coco_dets.append(detection)
#
#         self.results.extend(coco_dets)
#         self.img_ids.append(img_id)
#
#     def summarize(self):
#         json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
#         coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
#         coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
#         coco_eval.params.imgIds = self.img_ids
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
#         self.results = []
#         self.img_ids = []
#         self.aps.append(coco_eval.stats[0])
#         return {'ap': coco_eval.stats[0]}
#
#
# Evaluator = Evaluator if cfg.segm_or_bbox == 'segm' else DetectionEvaluator
