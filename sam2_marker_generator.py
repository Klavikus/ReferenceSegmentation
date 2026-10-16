from typing import Tuple, Optional

import cv2
import numpy as np


class SAM2MarkerGenerator:
    def __init__(
        self,
        model_size: str = "small",
        min_mask_area: int = 20,
        max_mask_area_ratio: float = 0.5,
        device: str = "auto",
    ):
        self.model_size = model_size
        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio
        self.device = device
        self._mask_generator = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            _ = SAM2AutomaticMaskGenerator
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _get_mask_generator(self):
        if self._mask_generator is not None:
            return self._mask_generator
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        model_id = f"facebook/sam2-hiera-{self.model_size}"
        self._mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
            model_id,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=self.min_mask_area,
            device=self.device,
        )
        return self._mask_generator

    def generate_markers(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        masks = self._generate_sam_masks(image)
        filtered = self._filter_masks(masks, (h, w))
        fg_mask, bg_mask = self._masks_to_markers(filtered, (h, w))
        return fg_mask, bg_mask

    def _generate_sam_masks(self, image: np.ndarray) -> list:
        generator = self._get_mask_generator()
        return generator.generate(image)

    def _filter_masks(self, masks: list, image_shape: tuple) -> list:
        h, w = image_shape
        total_area = h * w
        filtered = []
        for m in masks:
            area = m.get("area", 0)
            iou = m.get("predicted_iou", 0)
            if area < self.min_mask_area:
                continue
            if area > self.max_mask_area_ratio * total_area:
                continue
            if iou < 0.7:
                continue
            filtered.append(m)
        filtered.sort(key=lambda m: m.get("area", 0), reverse=True)
        filtered = self._nms(filtered, iou_threshold=0.8)
        return filtered

    def _nms(self, masks: list, iou_threshold: float = 0.8) -> list:
        if not masks:
            return []
        keep = []
        for i, m_i in enumerate(masks):
            seg_i = m_i["segmentation"].astype(bool)
            suppressed = False
            for m_j in keep:
                seg_j = m_j["segmentation"].astype(bool)
                intersection = np.logical_and(seg_i, seg_j).sum()
                union = np.logical_or(seg_i, seg_j).sum()
                if union == 0:
                    continue
                iou = intersection / union
                if iou > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(m_i)
        return keep

    def _masks_to_markers(
        self,
        filtered_masks: list,
        image_shape: tuple,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_shape
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for m in filtered_masks:
            seg = m["segmentation"].astype(np.uint8) * 255
            eroded = cv2.erode(seg, kernel_erosion, iterations=2)
            fg_mask = cv2.bitwise_or(fg_mask, eroded)
            combined_mask = cv2.bitwise_or(combined_mask, seg)

        dilated = cv2.dilate(combined_mask, kernel_dilation, iterations=5)
        bg_mask = cv2.bitwise_not(dilated)

        corner_size = 20
        bg_mask[0:corner_size, 0:corner_size] = 255
        bg_mask[0:corner_size, w - corner_size : w] = 255
        bg_mask[h - corner_size : h, 0:corner_size] = 255
        bg_mask[h - corner_size : h, w - corner_size : w] = 255

        return fg_mask, bg_mask


def build_sam2_marker_generator(**kwargs) -> SAM2MarkerGenerator:
    return SAM2MarkerGenerator(**kwargs)
