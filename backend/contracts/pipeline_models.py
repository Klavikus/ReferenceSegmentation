from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.core.segmentation import SegmentationResult


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class ImageOffset:
    x: int
    y: int


@dataclass(frozen=True)
class PreparedImageResult:
    prepared_image: np.ndarray
    grayscale_preview: np.ndarray
    prepared_size: ImageSize
    prepared_offset: ImageOffset
    scale_factor: float


@dataclass(frozen=True)
class PipelineConfig:
    target_width: int
    target_height: int
    default_pixel_to_nm: float = 1.0
    marker_generation_strategy: str = "sam2_optional"
    export_storage_kind: str = "local_temp_dir"
    contour_archive_name: str = "contours.zip"
    pipeline_version: str = "v1"

    @classmethod
    def standard(cls) -> "PipelineConfig":
        return cls(target_width=512, target_height=512)


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    prepared_image: np.ndarray
    grayscale_preview: np.ndarray
    prepared_size: ImageSize
    prepared_offset: ImageOffset
    scale_factor: float
    original_width: int
    original_height: int
    input_shape: list[int]
    input_dtype: str
    input_sha256: str
    input_role: str | None = None


@dataclass(frozen=True)
class PipelineManifest:
    pipeline_version: str
    model_version: str
    config_hash: str
    config: PipelineConfig
    input_roles: list[str]
    samples: list["ManifestSample"] = field(default_factory=list)


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    run_id: str | None
    input_role: str | None
    original_width: int
    original_height: int
    input_shape: list[int]
    input_dtype: str
    input_sha256: str
    marker_source: str | None = None
    pixel_to_nm: float | None = None
    artifacts: list["ExportArtifact"] = field(default_factory=list)


@dataclass(frozen=True)
class MarkerSuggestion:
    sample_id: str
    foreground_mask: np.ndarray
    background_mask: np.ndarray
    foreground_payload: dict
    background_payload: dict


@dataclass(frozen=True)
class ExportArtifact:
    artifact_type: str
    path: Path | str
    storage_kind: str = "local_temp_dir"
    filename: str | None = None
    mime_type: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        if self.filename is None:
            object.__setattr__(self, "filename", Path(self.path).name)


@dataclass(frozen=True)
class SegmentationExport:
    object_table: object
    object_table_path: str
    overall_table: object
    overall_table_path: str
    artifacts: list[ExportArtifact] = field(default_factory=list)


@dataclass(frozen=True)
class SampleSegmentationRun:
    run_id: str
    sample_id: str
    marker_source: str
    pixel_to_nm: float
    segmentation: SegmentationResult
    export: SegmentationExport
