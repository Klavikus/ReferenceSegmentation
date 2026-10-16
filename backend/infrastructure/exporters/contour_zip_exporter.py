import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ContourZipExportResult:
    zip_path: str


def render_contour_images(
    contours_original: list[np.ndarray],
    original_height: int,
    original_width: int,
    thickness: int,
) -> list[tuple[str, np.ndarray]]:
    contour_images = []
    for idx, contour in enumerate(contours_original):
        contour_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0, 255), thickness)
        contour_image_rgba = contour_image[:, :, [2, 1, 0, 3]]
        contour_images.append((f"contour_{idx + 1}.png", contour_image_rgba))
    return contour_images


def write_contour_zip(
    contours_original: list[np.ndarray],
    original_height: int,
    original_width: int,
    thickness: int,
    output_dir: str | Path,
) -> ContourZipExportResult:
    output_path = Path(output_dir)
    zip_path = output_path / "contours.zip"
    contour_images = render_contour_images(
        contours_original=contours_original,
        original_height=original_height,
        original_width=original_width,
        thickness=thickness,
    )

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for filename, contour_image_rgba in contour_images:
            temp_img_path = output_path / filename
            Image.fromarray(contour_image_rgba, "RGBA").save(temp_img_path)
            zipf.write(temp_img_path, arcname=filename)

    if not os.path.isfile(zip_path):
        raise ValueError("Error saving contour ZIP file")

    return ContourZipExportResult(zip_path=str(zip_path))
