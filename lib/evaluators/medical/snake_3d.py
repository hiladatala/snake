import os
import numpy as np
from dicom.my_dicom import myDicom
from lib.image.my_image import myImage
from collections import defaultdict

dicom = myDicom()
image = myImage()

class Evaluator:
    def __init__(self,dataset,result_dir):
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.dataset = dataset  # dataset info for GT volumes and slices
        self.volume_preds = {}  # {volume_id: list of slice preds}
        self.volume_gts = {}    # {volume_id: list of GT slices}

        self.metrics = []


    def evaluate_slice(self, pred_mask, gt_mask, volume_id, slice_idx):
        """
        Save predicted and GT slices for later 3D evaluation.
        pred_mask, gt_mask: 2D numpy arrays for one slice
        volume_id: unique ID for the volume (CT scan)
        slice_idx: index of this slice in the volume
        """
        if volume_id not in self.volume_preds:
            self.volume_preds[volume_id] = {}
            self.volume_gts[volume_id] = {}

        self.volume_preds[volume_id][slice_idx] = pred_mask
        self.volume_gts[volume_id][slice_idx] = gt_mask


    def Dice_metric(self, pred_volume, gt_volume):
        """
        Compute volumetric metrics like 3D Dice coefficient.
        """
        intersection = np.sum(pred_volume * gt_volume)
        pred_sum = np.sum(pred_volume)
        gt_sum = np.sum(gt_volume)
        dice = 2 * intersection / (pred_sum + gt_sum + 1e-6)
        return dice


    def summarize(self):
        """
        Stack slices into volumes and compute metrics.
        """
        for volume_id in self.volume_preds.keys():
            # get sorted slices by slice index
            slice_indices = sorted(self.volume_preds[volume_id].keys())
            pred_slices = [self.volume_preds[volume_id][idx] for idx in slice_indices]
            gt_slices = [self.volume_gts[volume_id][idx] for idx in slice_indices]

            pred_volume = np.stack(pred_slices, axis=0)
            gt_volume = np.stack(gt_slices, axis=0)

            dice_score = self.Dice_metric(pred_volume, gt_volume)
            self.metrics.append({'volume_id': volume_id, 'dice': dice_score})

            print(f"Volume {volume_id} Dice: {dice_score:.4f}")

        avg_dice = np.mean([m['dice'] for m in self.metrics])
        print(f"Average 3D Dice over all volumes: {avg_dice:.4f}")
        return avg_dice

class DetectionEvaluator:
    def __init__(self, result_dir, image_size=(512, 512)):
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        # Store predictions and GT per volume
        self.image_size = image_size
        self.volume_predictions = defaultdict(list)  # {volume_id: list of (slice_idx, boxes)}
        self.volume_gts = defaultdict(list)  # {volume_id: list of (slice_idx, gt_boxes)}

        self.metrics = []

    def evaluate_slice(self, output, batch):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        if detection.shape[0] == 0:
            return

        box = detection[:, :4].detach().cpu().numpy()  # [x1, y1, x2, y2]
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)

        meta = batch['meta']
        volume_id = meta['volume_id'][0]  # custom key for 3D volume identity
        slice_idx = int(meta['slice_idx'][0])
        # Optional: affine transform to map to original space if needed

        self.volume_predictions[volume_id].append((slice_idx, box, score, label))

        # Load GT boxes per slice
        gt_boxes = meta['gt_boxes'][0].detach().cpu().numpy()  # shape [N, 4]
        gt_labels = meta['gt_labels'][0].detach().cpu().numpy()
        self.volume_gts[volume_id].append((slice_idx, gt_boxes, gt_labels))

    def summarize(self):
        for volume_id in self.volume_predictions:
            pred_slices = self.volume_predictions[volume_id]
            gt_slices = self.volume_gts[volume_id]

            # Convert list of slices into 3D volume representation
            pred_volume_boxes = self.reconstruct_3d_boxes(pred_slices)
            gt_volume_boxes = self.reconstruct_3d_boxes(gt_slices, is_gt=True)

            # Match predictions to GT boxes in 3D and compute metrics
            dice, iou, centroid_error = self.evaluate_3d_detection(pred_volume_boxes, gt_volume_boxes)
            self.metrics.append({'volume_id': volume_id, 'dice': dice, 'iou': iou, 'centroid_error': centroid_error})

            print(f"[{volume_id}] 3D IoU: {iou:.4f}, Dice: {dice:.4f}, Centroid error: {centroid_error:.2f} mm")

        avg_iou = np.mean([m['iou'] for m in self.metrics])
        avg_dice = np.mean([m['dice'] for m in self.metrics])
        print(f"\nAverage 3D IoU: {avg_iou:.4f}, Average 3D Dice: {avg_dice:.4f}")
        return {'iou': avg_iou, 'dice': avg_dice}

    def reconstruct_3d_boxes(self, slice_boxes, is_gt=False):
        """
        Converts 2D bounding boxes on slices into a 3D binary volume.
        """
        if not slice_boxes:
            return np.zeros((1, *self.image_size), dtype=np.uint8)

        depth = max(s[0] for s in slice_boxes) + 1
        H, W = self.image_size
        volume = np.zeros((depth, H, W), dtype=np.uint8)

        for slice_idx, boxes, *rest in slice_boxes:
            for box in boxes:
                x1, y1, x2, y2 = np.round(box).astype(int)
                # Clip to volume boundaries
                x1, x2 = np.clip([x1, x2], 0, W)
                y1, y2 = np.clip([y1, y2], 0, H)
                if x1 < x2 and y1 < y2:
                    volume[slice_idx, y1:y2, x1:x2] = 1

        return volume

    def evaluate_3d_detection(self, pred_volume, gt_volume):
        intersection = np.sum(pred_volume * gt_volume)
        union = np.sum(np.clip(pred_volume + gt_volume, 0, 1))
        pred_sum = np.sum(pred_volume)
        gt_sum = np.sum(gt_volume)

        dice = 2 * intersection / (pred_sum + gt_sum + 1e-5) if (pred_sum + gt_sum) > 0 else 0.0
        iou = intersection / (union + 1e-5) if union > 0 else 0.0

        def get_centroid(volume):
            pos = np.argwhere(volume)
            return np.mean(pos, axis=0) if pos.size > 0 else None

        centroid_pred = get_centroid(pred_volume)
        centroid_gt = get_centroid(gt_volume)

        if centroid_pred is None or centroid_gt is None:
            centroid_error = float('nan')
        else:
            centroid_error = np.linalg.norm(centroid_pred - centroid_gt)

        return dice, iou, centroid_error

# Evaluator = Evaluator if cfg.segm_or_bbox == 'segm' else DetectionEvaluator
