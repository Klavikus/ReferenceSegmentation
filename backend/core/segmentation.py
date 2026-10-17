from dataclasses import asdict, dataclass

import cv2
import numpy as np

DEFAULT_PERCENTILES = [25, 50, 75, 95]
COLOR_BINS = 4
ALPHA_BLEND = 0.35
COLORS = [[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0]]


@dataclass(frozen=True)
class SegmentationObjectRow:
    ID: int
    area_px2: float
    perimeter_px: float
    center_x_px: int
    center_y_px: int
    area_nm2: float
    perimeter_nm: float
    center_x_nm: float
    center_y_nm: float

    def to_record(self) -> dict:
        record = asdict(self)
        return {
            "ID": record["ID"],
            "Площадь (px^2)": record["area_px2"],
            "Периметр (px)": record["perimeter_px"],
            "Центр X (px)": record["center_x_px"],
            "Центр Y (px)": record["center_y_px"],
            "Площадь (nm^2)": record["area_nm2"],
            "Периметр (nm)": record["perimeter_nm"],
            "Центр X (nm)": record["center_x_nm"],
            "Центр Y (nm)": record["center_y_nm"],
        }


@dataclass(frozen=True)
class SegmentationResult:
    object_rows: list[SegmentationObjectRow]
    contour_preview: np.ndarray
    percentile_preview: np.ndarray
    normalized_preview: np.ndarray
    contours_original: list[np.ndarray]


def segment_prepared_image(
    prepared_rgb: np.ndarray,
    foreground_mask: np.ndarray,
    background_mask: np.ndarray,
    pixel_to_nm: float,
    prepared_offset_x: int,
    prepared_offset_y: int,
    prepared_width: int,
    prepared_height: int,
    scale_factor: float,
    original_width: int,
    original_height: int,
) -> SegmentationResult:
    if pixel_to_nm <= 0.0:
        raise ValueError("pixel_to_nm should be more than 0.0")

    sure_fg = cv2.threshold(foreground_mask, 0, 255, cv2.THRESH_BINARY)[1]
    sure_bg = cv2.threshold(background_mask, 0, 255, cv2.THRESH_BINARY)[1]

    markers = np.zeros_like(sure_fg, dtype=np.int32)
    markers[sure_bg > 0] = 1
    markers[sure_fg > 0] = 255

    content_mask = np.zeros(prepared_rgb.shape[:2], dtype=bool)
    content_mask[
        prepared_offset_y : prepared_offset_y + prepared_height,
        prepared_offset_x : prepared_offset_x + prepared_width,
    ] = True
    markers[~content_mask] = 1

    image_bgr = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2BGR)
    watershed_markers = cv2.watershed(image_bgr, markers)
    result = image_bgr.copy()

    unique_markers = np.unique(watershed_markers)
    unique_markers = unique_markers[(unique_markers != 1) & (unique_markers != -1)]
    object_rows, contour_preview, percentile_preview, normalized_preview, contours = (
        _calculate_object_properties(
            unique_markers=unique_markers,
            markers=watershed_markers,
            result=result,
            image_bgr=image_bgr,
            pixel_to_nm=pixel_to_nm,
            prepared_offset_x=prepared_offset_x,
            prepared_offset_y=prepared_offset_y,
            scale_factor=scale_factor,
            original_width=original_width,
            original_height=original_height,
        )
    )

    if not object_rows:
        raise ValueError("No objects found")

    return SegmentationResult(
        object_rows=object_rows,
        contour_preview=contour_preview,
        percentile_preview=percentile_preview,
        normalized_preview=normalized_preview,
        contours_original=contours,
    )


def _calculate_object_properties(
    unique_markers: np.ndarray,
    markers: np.ndarray,
    result: np.ndarray,
    image_bgr: np.ndarray,
    pixel_to_nm: float,
    prepared_offset_x: int,
    prepared_offset_y: int,
    scale_factor: float,
    original_width: int,
    original_height: int,
) -> tuple[
    list[SegmentationObjectRow], np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]
]:
    object_rows: list[SegmentationObjectRow] = []
    contours_original: list[np.ndarray] = []
    result_contours = result.copy()
    result_colored_percentile = result.copy()
    result_colored_normalized = result.copy()

    # Калибровка задана для исходного изображения, измерения — после resize.
    pixel_to_nm /= scale_factor

    for marker in unique_markers:
        obj_mask = np.uint8(markers == marker)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            obj_mask, connectivity=8
        )

        for label in range(1, num_labels):
            component_mask = np.uint8(labels == label)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = contours[0]
            moments = cv2.moments(contour)
            area_px2 = float(np.sum(component_mask))
            perimeter = float(cv2.arcLength(contour, True))
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                center_x, center_y = 0, 0

            center_x_adjusted = center_x - prepared_offset_x
            center_y_adjusted = center_y - prepared_offset_y
            object_rows.append(
                SegmentationObjectRow(
                    ID=len(object_rows) + 1,
                    area_px2=area_px2,
                    perimeter_px=perimeter,
                    center_x_px=center_x_adjusted,
                    center_y_px=center_y_adjusted,
                    area_nm2=area_px2 * (pixel_to_nm**2),
                    perimeter_nm=perimeter * pixel_to_nm,
                    center_x_nm=center_x_adjusted * pixel_to_nm,
                    center_y_nm=center_y_adjusted * pixel_to_nm,
                )
            )

            adjusted_contour = []
            for point in contour:
                x_padded = point[0][0]
                y_padded = point[0][1]
                x_resized = x_padded - prepared_offset_x
                y_resized = y_padded - prepared_offset_y
                x_original = int(round(x_resized / scale_factor))
                y_original = int(round(y_resized / scale_factor))
                x_original = min(max(x_original, 0), original_width - 1)
                y_original = min(max(y_original, 0), original_height - 1)
                adjusted_contour.append([[x_original, y_original]])

            contours_original.append(np.array(adjusted_contour, dtype=np.int32))

            mask = labels == label
            object_image = np.zeros_like(image_bgr)
            object_image[mask] = image_bgr[mask]
            colored_object_percentile = _coloring_with_percentile(object_image, mask)
            colored_object_normalized = _coloring_with_normalized(object_image, mask)

            result_colored_percentile[mask] = cv2.addWeighted(
                result_colored_percentile[mask],
                1 - ALPHA_BLEND,
                colored_object_percentile[mask],
                ALPHA_BLEND,
                0,
            )
            result_colored_normalized[mask] = cv2.addWeighted(
                result_colored_normalized[mask],
                1 - ALPHA_BLEND,
                colored_object_normalized[mask],
                ALPHA_BLEND,
                0,
            )

            cv2.drawContours(result_contours, [contour], -1, (0, 0, 255), 1)
            cv2.circle(result_contours, (center_x, center_y), 3, (0, 255, 0), -1)

    return (
        object_rows,
        result_contours,
        result_colored_percentile,
        result_colored_normalized,
        contours_original,
    )


def _coloring_with_percentile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    percentiles = np.append(
        np.percentile(gray[mask > 0], DEFAULT_PERCENTILES), np.max(gray[mask > 0])
    )
    return _apply_coloring(gray, mask, percentiles)


def _coloring_with_normalized(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val = np.min(gray[mask > 0]), np.max(gray[mask > 0])
    normalized = np.clip((gray - min_val) / (max_val - min_val + 1e-7), 0, 1)
    thresholds = np.linspace(0, 1, COLOR_BINS + 1)
    return _apply_coloring(normalized, mask, thresholds)


def _apply_coloring(
    data: np.ndarray, mask: np.ndarray, boundaries: np.ndarray
) -> np.ndarray:
    colored = np.zeros((*data.shape, 3), dtype=np.uint8)
    for index in range(len(boundaries) - 1):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        interval_mask = (
            (data >= lower)
            & (
                (data < upper)
                if index < len(boundaries) - 2
                else (data <= upper)
            )
            & (mask > 0)
        )
        colored[interval_mask] = COLORS[index]
    return colored
