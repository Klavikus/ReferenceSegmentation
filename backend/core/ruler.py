from typing import Tuple

import numpy as np


def measure_ruler_distance(
    layer: np.ndarray,
    ruler_size_nm: float,
) -> Tuple[int, float]:
    red_pixels = layer[:, :, 0] > 100
    red_pixel_coords = np.where(red_pixels)
    if red_pixel_coords[1].size == 0:
        return 0, 0.0

    min_x = int(np.min(red_pixel_coords[1]))
    max_x = int(np.max(red_pixel_coords[1]))
    distance_px = max_x - min_x

    px_to_nm = 0.0
    if ruler_size_nm > 0 and distance_px > 0:
        px_to_nm = ruler_size_nm / distance_px

    return distance_px, px_to_nm
