from collections import defaultdict
import cv2
import numpy as np


def get_iou(intersection, pixel_sum):
    # handle edge cases without resorting to epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        assert intersection == 0
        return 1

    return intersection / (pixel_sum - intersection)

class CIoUEvaluator:
    def __init__(self):
        self.objects_in_gt = set()
        self.objects_in_masks = set()

        self.object_iou = defaultdict(list)
        self.frame_fg_object_iou = []

    def feed_frame(self, mask: np.ndarray, gt: np.ndarray, frame, object_id):
        """
        Compute and accumulate metrics for a single frame (mask/gt pair)
        """
        # # get all objects in the ground-truth
        # gt_objects = np.unique(gt)
        # gt_objects = gt_objects[gt_objects != 0].tolist()

        # # get all objects in the predicted mask
        # mask_objects = np.unique(mask)
        # mask_objects = mask_objects[mask_objects != 0].tolist()

        # self.objects_in_gt.update(set(gt_objects))
        # self.objects_in_masks.update(set(mask_objects))

        # all_objects = self.objects_in_gt.union(self.objects_in_masks)

        obj_mask = mask == object_id
        obj_gt = gt == object_id

        iou_value = get_iou((obj_mask * obj_gt).sum(), obj_mask.sum() + obj_gt.sum())
        # For challenge IoU, we only consider the object in the ground-truth
        if np.sum(obj_gt) != 0:
            self.frame_fg_object_iou.append((frame, object_id, iou_value))



