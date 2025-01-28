from pathlib import Path
from typing import Any
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from numpy import ndarray

STANDARD_WIDTH = 512
STANDARD_HEIGHT = 512
DEFAULT_PERCENTILES = [25, 50, 75, 95]
COLOR_BINS = 4
ALPHA_BLEND = 0.35
COLORS = [
    [255, 0, 0],
    [255, 165, 0],
    [255, 255, 0],
    [0, 255, 0]
]


class WatershedProcessor:
    def __init__(self, width: int = STANDARD_WIDTH, height: int = STANDARD_HEIGHT, pixel_to_nm: float = 1.0):
        self.width = width
        self.height = height
        self.pixel_to_nm = pixel_to_nm
        self.original_height = None
        self.original_width = None
        self.scale_factor = None
        self.prepared_x_offset = None
        self.prepared_y_offset = None
        self.contours_original = []

    def prepare_image(self, image: np.ndarray) -> ndarray:
        if image is None:
            raise ValueError("Image is None")
        self.original_height = image.shape[0]
        self.original_width = image.shape[1]
        resized_image, x_offset, y_offset = self.__preprocess_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.prepared_x_offset = x_offset
        self.prepared_y_offset = y_offset
        h, w = image.shape[:2]
        self.scale_factor = min(self.width / w, self.height / h)
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def watershed_algorithm(self, image: np.ndarray, mask_data_f: dict, mask_data_b: dict, pixel_to_nm: float) -> tuple:
        if pixel_to_nm == 0.0:
            raise ValueError("pixel_to_nm should be more than 0.0")
        if image is None or mask_data_f is None or mask_data_b is None:
            raise ValueError("Missing image or mask data")
        if 'layers' not in mask_data_f or not mask_data_f['layers']:
            raise ValueError("No Mask F layers provided")
        if 'layers' not in mask_data_b or not mask_data_b['layers']:
            raise ValueError("No Mask B layers provided")
        self.pixel_to_nm = pixel_to_nm
        mask_sf = np.array(mask_data_f['layers'][0])
        mask_sb = np.array(mask_data_b['layers'][0])
        mask_sf = cv2.cvtColor(mask_sf, cv2.COLOR_RGBA2GRAY)
        mask_sb = cv2.cvtColor(mask_sb, cv2.COLOR_RGBA2GRAY)
        mask_sf, x_offset, y_offset = self.__preprocess_image(mask_sf)
        mask_sb, _, _ = self.__preprocess_image(mask_sb)

        self.contours_original = []

        _, sure_fg = cv2.threshold(mask_sf, 0, 255, cv2.THRESH_BINARY)
        _, sure_bg = cv2.threshold(mask_sb, 0, 255, cv2.THRESH_BINARY)

        markers = np.zeros_like(sure_fg, dtype=np.int32)
        markers[sure_bg > 0] = 1
        markers[sure_fg > 0] = 255

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(image_bgr, markers)
        result = image_bgr.copy()

        unique_markers = np.unique(markers)
        unique_markers = unique_markers[(unique_markers != 1) & (unique_markers != -1)]
        data, result_contours, result_colored_percentile, result_colored_normalized = self.__calculate_object_properties(
            unique_markers, markers, result,
            image_bgr, x_offset,
            y_offset)

        if not data:
            raise ValueError("No objects found")

        df = pd.DataFrame(data)
        excel_path = "output.xlsx"
        df.to_excel(excel_path, index=False, sheet_name='Сегментированные объекты')

        if not Path(excel_path).is_file():
            raise ValueError("Error saving Excel file")

        excel_path_overall = "overall.xlsx"

        overall_data = {
            'Общее кол-во частиц': [len(data)],
            'Средняя площадь (px^2)': [df['Площадь (px^2)'].mean()],
            'Средний периметр (px)': [df['Периметр (px)'].mean()],
            'Средняя площадь (nm^2)': [df['Площадь (nm^2)'].mean()],
            'Средний периметр (nm)': [df['Периметр (nm)'].mean()]
        }
        df_overall = pd.DataFrame(overall_data)
        df_overall.to_excel(excel_path_overall, index=False, sheet_name='Общие данные')

        if not Path(excel_path_overall).is_file():
            raise ValueError("Error saving overall Excel file")

        return result_contours, result_colored_percentile, result_colored_normalized, df, excel_path, df_overall, excel_path_overall

    def __preprocess_image(self, image: np.ndarray) -> Any:
        if image is None:
            raise ValueError("Image is None")
        h, w = image.shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        x_offset = (self.width - new_w) // 2
        y_offset = (self.height - new_h) // 2
        padded = self.__resize_and_pad_image(resized, x_offset, y_offset)
        return padded, x_offset, y_offset

    def __resize_and_pad_image(self, image: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
        if len(image.shape) == 2:
            padded = np.full((self.height, self.width), 255, dtype=image.dtype)
            padded[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
        else:
            padded = np.full((self.height, self.width, 3), 255, dtype=image.dtype)
            padded[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1], :] = image
        return padded

    def __calculate_object_properties(self, unique_markers, markers, result, image_bgr, x_offset, y_offset) -> tuple:
        data = []
        current_object_id = 1
        result_contours = result.copy()
        result_colored_percentile = result.copy()
        result_colored_normalized = result.copy()

        for marker in unique_markers:
            obj_mask = np.uint8(markers == marker)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)

            for label in range(1, num_labels):  # label=0 это фон внутри маски
                component_mask = np.uint8(labels == label)
                contours, _ = cv2.findContours(component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    contour = contours[0]
                    m = cv2.moments(contour)
                    area = np.sum(component_mask) * self.pixel_to_nm
                    perimeter = cv2.arcLength(contour, True)
                    if m["m00"] != 0:
                        c_x = int(m["m10"] / m["m00"])
                        c_y = int(m["m01"] / m["m00"])
                    else:
                        c_x, c_y = 0, 0

                    c_x_adj = c_x - x_offset
                    c_y_adj = c_y - y_offset

                    data.append({
                        "ID": len(data) + 1,
                        "Площадь (px^2)": area,
                        "Периметр (px)": perimeter,
                        "Центр X (px)": c_x_adj,
                        "Центр Y (px)": c_y_adj,
                        "Площадь (nm^2)": area * self.pixel_to_nm,
                        "Периметр (nm)": perimeter * self.pixel_to_nm,
                        "Центр X (nm)": c_x_adj * self.pixel_to_nm,
                        "Центр Y (nm)": c_y_adj * self.pixel_to_nm
                    })

                    adjusted_contour = []
                    for point in contour:
                        x_padded = point[0][0]
                        y_padded = point[0][1]
                        x_resized = x_padded - self.prepared_x_offset
                        y_resized = y_padded - self.prepared_y_offset
                        x_original = int(round(x_resized / self.scale_factor))
                        y_original = int(round(y_resized / self.scale_factor))
                        adjusted_contour.append([[x_original, y_original]])

                    adjusted_contour_np = np.array(adjusted_contour, dtype=np.int32)
                    self.contours_original.append(adjusted_contour_np)

                    mask = labels == label
                    object_image = np.zeros_like(image_bgr)
                    object_image[mask] = image_bgr[mask]
                    colored_object_percentile = self.__coloring_with_percentile(object_image, mask)
                    colored_object_normalized = self.__coloring_with_normalized(object_image, mask)
                    alpha = 0.35

                    result_colored_percentile[mask] = cv2.addWeighted(result_colored_percentile[mask], 1 - alpha,
                                                                      colored_object_percentile[mask], alpha,
                                                                      0)
                    result_colored_normalized[mask] = cv2.addWeighted(result_colored_normalized[mask], 1 - alpha,
                                                                      colored_object_normalized[mask], alpha,
                                                                      0)

                    cv2.drawContours(result_contours, [contour], -1, (0, 0, 255), 1)
                    cv2.circle(result_contours, (c_x, c_y), 3, (0, 255, 0), -1)
                    current_object_id += 1
        return data, result_contours, result_colored_percentile, result_colored_normalized

    def __coloring_with_percentile(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        percentiles = np.append(np.percentile(gray[mask > 0], DEFAULT_PERCENTILES), np.max(gray[mask > 0]))
        return self.__apply_coloring(gray, mask, percentiles)

    def __coloring_with_normalized(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_val, max_val = np.min(gray[mask > 0]), np.max(gray[mask > 0])
        normalized = np.clip((gray - min_val) / (max_val - min_val + 1e-7), 0, 1)
        thresholds = np.linspace(0, 1, COLOR_BINS + 1)
        return self.__apply_coloring(normalized, mask, thresholds)

    def __apply_coloring(self, data: np.ndarray, mask: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
        colored = np.zeros((*data.shape, 3), dtype=np.uint8)
        for i in range(len(boundaries) - 1):
            lower, upper = boundaries[i], boundaries[i + 1]
            interval_mask = (data >= lower) & ((data < upper) if i < len(boundaries) - 2 else (data <= upper)) & (
                    mask > 0)
            colored[interval_mask] = COLORS[i]
        return colored

    def generate_contour_images(self, thickness: int) -> List[Tuple[str, np.ndarray]]:
        if self.original_height is None or self.original_width is None:
            return []
        contour_images = []
        for idx, contour in enumerate(self.contours_original):
            contour_image = np.zeros((self.original_height, self.original_width, 4), dtype=np.uint8)
            cv2.drawContours(contour_image, [contour], -1, (0, 255, 0, 255), thickness)
            contour_image_rgba = contour_image[:, :, [2, 1, 0, 3]]
            contour_images.append((f"contour_{idx + 1}.png", contour_image_rgba))
        return contour_images
