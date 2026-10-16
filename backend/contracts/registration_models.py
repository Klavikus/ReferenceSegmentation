from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RegistrationConfig:
    proxy_max_side: int = 1600
    sift_nfeatures: int = 4000
    lowe_ratio: float = 0.75
    ransac_reproj_threshold: float = 3.0
    min_good_matches: int = 10
    scale_warning_threshold: float = 0.02
    ecc_max_iterations: int = 100
    ecc_epsilon: float = 1e-6
    ecc_gauss_filter_size: int = 5
    use_clahe: bool = True
    bottom_exclusion_px: int = 120


@dataclass(frozen=True)
class CropRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class RegistrationDiagnostics:
    rotation_deg: float
    translation_x_px: float
    translation_y_px: float
    scale_hint: float | None
    keypoints_before: int
    keypoints_after: int
    good_matches: int
    inliers: int
    inlier_ratio: float | None
    ecc_score: float | None
    overlap_ratio: float
    bottom_exclusion_px: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rotation_deg": self.rotation_deg,
            "translation_x_px": self.translation_x_px,
            "translation_y_px": self.translation_y_px,
            "scale_applied": 1.0,
            "scale_hint": self.scale_hint,
            "keypoints_before": self.keypoints_before,
            "keypoints_after": self.keypoints_after,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "ecc_score": self.ecc_score,
            "overlap_ratio": self.overlap_ratio,
            "bottom_exclusion_px": self.bottom_exclusion_px,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RegistrationResult:
    before_aligned: np.ndarray
    after_aligned: np.ndarray
    overlay_preview: np.ndarray
    transform_matrix: np.ndarray
    crop_rect: CropRect
    diagnostics: RegistrationDiagnostics


@dataclass(frozen=True)
class RegistrationExport:
    output_dir: Path
    before_path: Path
    after_path: Path
    overlay_path: Path
    metadata_path: Path
    zip_path: Path


@dataclass(frozen=True)
class RegistrationRun:
    result: RegistrationResult
    export: RegistrationExport
