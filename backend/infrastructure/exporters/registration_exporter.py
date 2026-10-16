import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from backend.contracts.registration_models import RegistrationExport, RegistrationResult


def _image_for_cv2(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    elif array.dtype not in (np.uint8, np.uint16):
        if np.issubdtype(array.dtype, np.floating):
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                raise ValueError("Невозможно сохранить изображение без конечных значений")
            if finite.min() >= 0.0 and finite.max() <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 65535).astype(np.uint16)

    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    return array


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), _image_for_cv2(image)):
        raise ValueError(f"Не удалось сохранить изображение: {path}")


def write_registration_export(
    result: RegistrationResult,
    output_dir: str | Path,
) -> RegistrationExport:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    before_path = output_path / "before_aligned.png"
    after_path = output_path / "after_aligned.png"
    overlay_path = output_path / "overlay.png"
    metadata_path = output_path / "registration.json"
    zip_path = output_path / "aligned_pair.zip"

    _write_png(before_path, result.before_aligned)
    _write_png(after_path, result.after_aligned)
    _write_png(overlay_path, result.overlay_preview)

    metadata = {
        "transform_model": "euclidean_rigid",
        "reference": "before",
        "moving": "after",
        "scale_applied": 1.0,
        "transform_after_to_before": result.transform_matrix.tolist(),
        "crop_rect_reference": {
            "x": result.crop_rect.x,
            "y": result.crop_rect.y,
            "width": result.crop_rect.width,
            "height": result.crop_rect.height,
        },
        "output_shape": list(result.before_aligned.shape),
        "diagnostics": result.diagnostics.to_dict(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (before_path, after_path, overlay_path, metadata_path):
            archive.write(path, arcname=path.name)

    return RegistrationExport(
        output_dir=output_path,
        before_path=before_path,
        after_path=after_path,
        overlay_path=overlay_path,
        metadata_path=metadata_path,
        zip_path=zip_path,
    )
