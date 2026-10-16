import hashlib
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import cv2
import numpy as np

from backend.contracts.pipeline_models import (
    ExportArtifact,
    MarkerSuggestion,
    ManifestSample,
    PipelineConfig,
    PipelineManifest,
    PreparedSample,
    SegmentationExport,
    SampleSegmentationRun,
)
from backend.core.mask_utils import extract_first_mask_layer, validate_mask_payload
from backend.core.preprocessing import prepare_image
from backend.core.segmentation import segment_prepared_image
from backend.infrastructure.exporters.contour_zip_exporter import write_contour_zip
from backend.infrastructure.exporters.excel_exporter import write_segmentation_excels


MARKER_MODE_FULL = "Полная заливка"
MARKER_MODE_CONTOUR = "Только контур"
MARKER_MODE_EMPTY = "Не заливать"
MARKER_MODES = {MARKER_MODE_FULL, MARKER_MODE_CONTOUR, MARKER_MODE_EMPTY}


class PipelineService:
    def __init__(
        self,
        config: PipelineConfig,
        segmenter=None,
        exporter=None,
        sam2_adapter=None,
    ):
        self.config = config
        self._prepared_samples: dict[str, PreparedSample] = {}
        self._runs: dict[str, SampleSegmentationRun] = {}
        self._latest_run_ids: dict[str, str] = {}
        self._segmenter = segmenter or segment_prepared_image
        self._exporter = exporter or write_segmentation_excels
        self._sam2_adapter = sam2_adapter

    def create_manifest(self, input_roles: list[str]) -> PipelineManifest:
        samples = list(self._prepared_samples.values())
        return PipelineManifest(
            pipeline_version=self.config.pipeline_version,
            model_version="watershed",
            config_hash=self._build_config_hash(),
            config=self.config,
            input_roles=input_roles,
            samples=[
                ManifestSample(
                    sample_id=sample.sample_id,
                    run_id=(latest_run.run_id if latest_run is not None else None),
                    input_role=sample.input_role
                    or (input_roles[index] if index < len(input_roles) else None),
                    original_width=sample.original_width,
                    original_height=sample.original_height,
                    input_shape=list(sample.input_shape),
                    input_dtype=sample.input_dtype,
                    input_sha256=sample.input_sha256,
                    marker_source=(latest_run.marker_source if latest_run is not None else None),
                    pixel_to_nm=(latest_run.pixel_to_nm if latest_run is not None else None),
                    artifacts=(list(latest_run.export.artifacts) if latest_run is not None else []),
                )
                for index, sample in enumerate(samples)
                for latest_run in [self._get_latest_run(sample.sample_id)]
            ],
        )

    def can_suggest_markers(self) -> bool:
        return self._sam2_adapter is not None and self._sam2_adapter.is_available()

    def prepare_sample(
        self,
        image: np.ndarray,
        *,
        replace_cache: bool = True,
        input_role: str | None = None,
    ) -> PreparedSample:
        if image is None:
            raise ValueError("Image is None")

        if replace_cache:
            self._prepared_samples.clear()
            self._runs.clear()
            self._latest_run_ids.clear()

        prepared_result = prepare_image(
            image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            target_width=self.config.target_width,
            target_height=self.config.target_height,
        )
        sample = PreparedSample(
            sample_id=str(uuid4()),
            input_role=input_role,
            prepared_image=prepared_result.prepared_image,
            grayscale_preview=prepared_result.grayscale_preview,
            prepared_size=prepared_result.prepared_size,
            prepared_offset=prepared_result.prepared_offset,
            scale_factor=prepared_result.scale_factor,
            original_width=image.shape[1],
            original_height=image.shape[0],
            input_shape=list(image.shape),
            input_dtype=str(image.dtype),
            input_sha256=self._build_input_hash(image),
        )
        self._prepared_samples[sample.sample_id] = sample
        return sample

    def prepare_batch_samples(self, images_by_role: dict[str, np.ndarray]) -> dict[str, PreparedSample]:
        self._prepared_samples.clear()
        self._runs.clear()
        self._latest_run_ids.clear()
        prepared_samples: dict[str, PreparedSample] = {}
        for role, image in images_by_role.items():
            prepared_samples[role] = self.prepare_sample(
                image,
                replace_cache=False,
                input_role=role,
            )
        return prepared_samples

    def suggest_markers(self, sample_id: str, image: np.ndarray) -> MarkerSuggestion:
        sample = self._require_sample(sample_id)
        if not self.can_suggest_markers():
            raise ValueError("SAM2 adapter is not available")

        foreground_mask, background_mask = self._sam2_adapter.generate_markers(
            sample.grayscale_preview
        )
        foreground_payload, background_payload = self.create_marker_payloads(
            sample_id=sample.sample_id,
            foreground_mask=foreground_mask,
            background_mask=background_mask,
        )
        return MarkerSuggestion(
            sample_id=sample.sample_id,
            foreground_mask=foreground_mask,
            background_mask=background_mask,
            foreground_payload=foreground_payload,
            background_payload=background_payload,
        )

    def segment_sample(
        self,
        sample_id: str,
        foreground_payload: dict,
        background_payload: dict,
        pixel_to_nm: float,
        marker_source: str = "manual",
    ) -> SampleSegmentationRun:
        if pixel_to_nm <= 0.0:
            raise ValueError("pixel_to_nm should be more than 0.0")

        sample = self._prepared_samples.get(sample_id)
        if sample is None:
            raise ValueError(f"Unknown sample_id: {sample_id}")

        validate_mask_payload(foreground_payload)
        validate_mask_payload(background_payload)

        foreground_mask = self._prepare_mask(extract_first_mask_layer(foreground_payload), sample)
        background_mask = self._prepare_mask(extract_first_mask_layer(background_payload), sample)

        segmentation = self._segmenter(
            prepared_rgb=sample.grayscale_preview,
            foreground_mask=foreground_mask,
            background_mask=background_mask,
            pixel_to_nm=pixel_to_nm,
            prepared_offset_x=sample.prepared_offset.x,
            prepared_offset_y=sample.prepared_offset.y,
            prepared_width=sample.prepared_size.width,
            prepared_height=sample.prepared_size.height,
            scale_factor=sample.scale_factor,
            original_width=sample.original_width,
            original_height=sample.original_height,
        )
        run_id = str(uuid4())
        export = self._exporter(
            rows=[object_row.to_record() for object_row in segmentation.object_rows],
            output_dir=self._create_run_output_dir(sample_id, run_id),
        )
        run = SampleSegmentationRun(
            run_id=run_id,
            sample_id=sample_id,
            marker_source=marker_source,
            pixel_to_nm=pixel_to_nm,
            segmentation=segmentation,
            export=export,
        )
        self._runs[run_id] = run
        self._latest_run_ids[sample_id] = run_id
        return run

    def create_marker_payloads(
        self,
        sample_id: str,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray,
        foreground_mode: str = MARKER_MODE_FULL,
        background_mode: str = MARKER_MODE_FULL,
        contour_thickness: int = 3,
    ) -> tuple[dict, dict]:
        sample = self._require_sample(sample_id)
        foreground_prepared = self._prepare_marker_payload_mask(foreground_mask, sample)
        background_prepared = self._prepare_marker_payload_mask(background_mask, sample)
        foreground_prepared = self._apply_marker_mode(
            foreground_prepared,
            foreground_mode,
            contour_thickness,
        )
        background_prepared = self._apply_marker_mode(
            background_prepared,
            background_mode,
            contour_thickness,
        )

        return (
            self._build_marker_payload(
                base_image=sample.grayscale_preview,
                mask=foreground_prepared,
                channel_index=2,
            ),
            self._build_marker_payload(
                base_image=sample.grayscale_preview,
                mask=background_prepared,
                channel_index=0,
            ),
        )

    def export_contours_zip(
        self,
        sample_id: str,
        contours_original: list[np.ndarray],
        thickness: int,
    ):
        sample = self._require_sample(sample_id)
        result = write_contour_zip(
            contours_original=contours_original,
            original_height=sample.original_height,
            original_width=sample.original_width,
            thickness=thickness,
            output_dir=tempfile.mkdtemp(),
        )
        existing_run = self._get_latest_run(sample_id)
        if existing_run is not None:
            contour_artifact = ExportArtifact(
                artifact_type="contour_zip",
                path=result.zip_path,
                mime_type="application/zip",
            )
            updated_export = SegmentationExport(
                object_table=existing_run.export.object_table,
                object_table_path=existing_run.export.object_table_path,
                overall_table=existing_run.export.overall_table,
                overall_table_path=existing_run.export.overall_table_path,
                artifacts=[*existing_run.export.artifacts, contour_artifact],
            )
            updated_run = SampleSegmentationRun(
                run_id=existing_run.run_id,
                sample_id=existing_run.sample_id,
                marker_source=existing_run.marker_source,
                pixel_to_nm=existing_run.pixel_to_nm,
                segmentation=existing_run.segmentation,
                export=updated_export,
            )
            self._runs[existing_run.run_id] = updated_run
        return result

    def _apply_marker_mode(
        self,
        mask: np.ndarray,
        mode: str,
        contour_thickness: int,
    ) -> np.ndarray:
        if mode not in MARKER_MODES:
            raise ValueError(f"Unknown marker mode: {mode}")

        if mode == MARKER_MODE_EMPTY:
            return np.zeros_like(mask)
        if mode == MARKER_MODE_FULL:
            return mask

        thickness_value = float(contour_thickness)
        if thickness_value < 1:
            raise ValueError("contour_thickness must be at least 1")
        if not thickness_value.is_integer():
            raise ValueError("contour_thickness must be an integer")
        thickness = int(thickness_value)

        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary_mask, kernel, iterations=thickness)
        return cv2.subtract(binary_mask, eroded)

    def _prepare_mask(self, mask: np.ndarray, sample: PreparedSample) -> np.ndarray:
        if mask.shape[:2] == (self.config.target_height, self.config.target_width):
            return mask
        return self._fit_mask_to_prepared_region(mask, sample)

    def _prepare_marker_payload_mask(
        self, mask: np.ndarray, sample: PreparedSample
    ) -> np.ndarray:
        if mask.shape[:2] == (self.config.target_height, self.config.target_width):
            prepared_mask = mask.copy()
        else:
            prepared_mask = self._fit_mask_to_prepared_region(mask, sample)
        return self._clear_mask_outside_prepared_region(prepared_mask, sample)

    def _clear_mask_outside_prepared_region(
        self, mask: np.ndarray, sample: PreparedSample
    ) -> np.ndarray:
        content_mask = np.zeros(mask.shape[:2], dtype=bool)
        content_mask[
            sample.prepared_offset.y : sample.prepared_offset.y + sample.prepared_size.height,
            sample.prepared_offset.x : sample.prepared_offset.x + sample.prepared_size.width,
        ] = True
        mask[~content_mask] = 0
        return mask

    def _fit_mask_to_prepared_region(
        self, mask: np.ndarray, sample: PreparedSample
    ) -> np.ndarray:
        resized_mask = cv2.resize(
            mask,
            (sample.prepared_size.width, sample.prepared_size.height),
        )
        padded_mask = np.zeros(
            (self.config.target_height, self.config.target_width),
            dtype=mask.dtype,
        )
        padded_mask[
            sample.prepared_offset.y : sample.prepared_offset.y + sample.prepared_size.height,
            sample.prepared_offset.x : sample.prepared_offset.x + sample.prepared_size.width,
        ] = resized_mask
        return padded_mask

    def _require_sample(self, sample_id: str) -> PreparedSample:
        sample = self._prepared_samples.get(sample_id)
        if sample is None:
            raise ValueError(f"Unknown sample_id: {sample_id}")
        return sample

    def _get_latest_run(self, sample_id: str) -> SampleSegmentationRun | None:
        run_id = self._latest_run_ids.get(sample_id)
        if run_id is None:
            return None
        return self._runs.get(run_id)

    def _build_marker_payload(
        self,
        base_image: np.ndarray,
        mask: np.ndarray,
        channel_index: int,
    ) -> dict:
        rgba = np.zeros((*base_image.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, channel_index] = mask
        rgba[:, :, 3] = mask
        background = np.ascontiguousarray(base_image.copy())
        return {
            "background": background,
            "layers": [rgba],
            "composite": background.copy(),
        }

    def _create_run_output_dir(self, sample_id: str, run_id: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"segmentation-{sample_id[:8]}-{run_id[:8]}-"))

    def _build_config_hash(self) -> str:
        payload = json.dumps(
            {
                "target_width": self.config.target_width,
                "target_height": self.config.target_height,
                "default_pixel_to_nm": self.config.default_pixel_to_nm,
                "marker_generation_strategy": self.config.marker_generation_strategy,
                "export_storage_kind": self.config.export_storage_kind,
                "contour_archive_name": self.config.contour_archive_name,
                "pipeline_version": self.config.pipeline_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_input_hash(self, image: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
