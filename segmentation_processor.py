from typing import Tuple, List

import cv2
import numpy as np
from backend.application.pipeline_service import PipelineService
from backend.contracts.pipeline_models import PipelineConfig, PreparedSample
from backend.core.preprocessing import resize_and_pad_image
from backend.core.segmentation import segment_prepared_image
from backend.infrastructure.exporters.contour_zip_exporter import render_contour_images
from backend.infrastructure.exporters.excel_exporter import write_segmentation_excels
from numpy import ndarray

STANDARD_WIDTH = 512
STANDARD_HEIGHT = 512


class WatershedProcessor:
    def __init__(
        self,
        width: int = STANDARD_WIDTH,
        height: int = STANDARD_HEIGHT,
        pixel_to_nm: float = 1.0,
        pipeline_service: PipelineService | None = None,
    ):
        self.width = width
        self.height = height
        self.pixel_to_nm = pixel_to_nm
        if pipeline_service is not None:
            config = pipeline_service.config
            if config.target_width != width or config.target_height != height:
                raise ValueError(
                    "PipelineService config dimensions must match WatershedProcessor dimensions"
                )
        self.pipeline_service = pipeline_service or PipelineService(
            config=PipelineConfig(target_width=width, target_height=height),
            segmenter=lambda **kwargs: segment_prepared_image(**kwargs),
            exporter=lambda **kwargs: write_segmentation_excels(**kwargs),
        )
        self.original_height = None
        self.original_width = None
        self.scale_factor = None
        self.prepared_x_offset = None
        self.prepared_y_offset = None
        self.prepared_new_w = None
        self.prepared_new_h = None
        self.contours_original = []
        self._prepared_sample: PreparedSample | None = None

    def prepare_image(self, image: np.ndarray) -> ndarray:
        prepared_sample = self.pipeline_service.prepare_sample(image)
        self._prepared_sample = prepared_sample
        self.original_height = prepared_sample.original_height
        self.original_width = prepared_sample.original_width
        self.prepared_x_offset = prepared_sample.prepared_offset.x
        self.prepared_y_offset = prepared_sample.prepared_offset.y
        self.scale_factor = prepared_sample.scale_factor
        self.prepared_new_w = prepared_sample.prepared_size.width
        self.prepared_new_h = prepared_sample.prepared_size.height
        return prepared_sample.grayscale_preview

    def fit_to_prepared_size(self, mask: np.ndarray) -> np.ndarray:
        if self.prepared_new_w is None or self.prepared_new_h is None:
            raise ValueError("Сначала вызовите prepare_image()")
        resized = cv2.resize(mask, (self.prepared_new_w, self.prepared_new_h))
        padded = np.zeros((self.height, self.width), dtype=np.uint8)
        padded[
            self.prepared_y_offset : self.prepared_y_offset + self.prepared_new_h,
            self.prepared_x_offset : self.prepared_x_offset + self.prepared_new_w,
        ] = resized
        return padded

    def watershed_algorithm(
        self,
        image: np.ndarray,
        mask_data_f: dict,
        mask_data_b: dict,
        pixel_to_nm: float,
    ) -> tuple:
        if self._prepared_sample is None:
            raise ValueError("Сначала вызовите prepare_image()")
        if image is None or mask_data_f is None or mask_data_b is None:
            raise ValueError("Missing image or mask data")
        self.pixel_to_nm = pixel_to_nm
        self.contours_original = []
        run = self.pipeline_service.segment_sample(
            sample_id=self._prepared_sample.sample_id,
            foreground_payload=mask_data_f,
            background_payload=mask_data_b,
            pixel_to_nm=self.pixel_to_nm,
        )
        self.contours_original = run.segmentation.contours_original

        return (
            run.segmentation.contour_preview,
            run.segmentation.percentile_preview,
            run.segmentation.normalized_preview,
            run.export.object_table,
            run.export.object_table_path,
            run.export.overall_table,
            run.export.overall_table_path,
        )

    def __preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        prepared_result = resize_and_pad_image(
            image=image,
            target_width=self.width,
            target_height=self.height,
        )
        return (
            prepared_result.prepared_image,
            prepared_result.prepared_offset.x,
            prepared_result.prepared_offset.y,
        )

    def generate_contour_images(self, thickness: int) -> List[Tuple[str, np.ndarray]]:
        if self.original_height is None or self.original_width is None:
            return []
        return render_contour_images(
            contours_original=self.contours_original,
            original_height=self.original_height,
            original_width=self.original_width,
            thickness=thickness,
        )
