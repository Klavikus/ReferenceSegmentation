from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy import ndarray

STANDARD_WIDTH = 512
STANDARD_HEIGHT = 512


class WatershedProcessor:
    def __init__(self, width: int = STANDARD_WIDTH, height: int = STANDARD_HEIGHT, pixel_to_nm: float = 1.0):
        self.width = width
        self.height = height
        self.pixel_to_nm = pixel_to_nm

    def prepare_image(self, image: np.ndarray) -> ndarray:
        if image is None:
            raise ValueError("Image is None")
        resized_image, _, _ = self.__preprocess_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)
        return gray_rgb

    def watershed_algorithm(self, image: np.ndarray, mask_data_f: dict, mask_data_b: dict) -> tuple:
        if image is None or mask_data_f is None or mask_data_b is None:
            raise ValueError("Missing image or mask data")
        if 'layers' not in mask_data_f or not mask_data_f['layers']:
            raise ValueError("No Mask F layers provided")
        if 'layers' not in mask_data_b or not mask_data_b['layers']:
            raise ValueError("No Mask B layers provided")
        mask_sf = np.array(mask_data_f['layers'][0])
        mask_sb = np.array(mask_data_b['layers'][0])
        mask_sf = cv2.cvtColor(mask_sf, cv2.COLOR_RGBA2GRAY)
        mask_sb = cv2.cvtColor(mask_sb, cv2.COLOR_RGBA2GRAY)
        mask_sf, x_offset, y_offset = self.__preprocess_image(mask_sf)
        mask_sb, _, _ = self.__preprocess_image(mask_sb)

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
        data, result_contours, result_colored_percentile, result_colored_normalized = self.__calculate_object_properties(unique_markers, markers, result,
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
        total_particles = len(data)
        avg_area = df['Площадь'].mean()
        avg_perimeter = df['Периметр'].mean()
        overall_data = {
            'Общее кол-во частиц': [total_particles],
            'Средняя площадь': [avg_area],
            'Средний периметр': [avg_perimeter]
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
                        "ID": current_object_id,
                        "Площадь": area,
                        "Периметр": perimeter,
                        "Центр X": c_x_adj,
                        "Центр Y": c_y_adj
                    })

                    mask = labels == label
                    object_image = np.zeros_like(image_bgr)
                    object_image[mask] = image_bgr[mask]
                    colored_object_percentile = self.__coloring_with_percentile(object_image, mask)
                    colored_object_normalized = self.__coloring_with_normilized(object_image, mask)
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

    def __coloring_with_percentile(self, image: np.ndarray, mask: bool) -> np.ndarray:
        if image is None:
            raise ValueError("Image is None")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        percentiles = np.percentile(gray[mask], [25, 50, 75, 95])
        colors = [[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0]]
        colored_image = np.zeros_like(image)
        for i in range(len(percentiles) - 1):
            range_mask = (gray >= percentiles[i]) & (gray < percentiles[i + 1])
            colored_image[range_mask] = colors[i]
        return colored_image

    def __coloring_with_normilized(self, image: np.ndarray, mask: bool) -> np.ndarray:
        if image is None:
            raise ValueError("Image is None")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        min_val = np.min(gray[mask])
        max_val = np.max(gray[mask])
        normalized = np.clip((gray - min_val) / (max_val - min_val), 0, 1)
        n = 4
        thresholds = np.linspace(0, 1, n + 1)
        colors = [[255, 0, 0], [255, 165, 0], [255, 255, 0], [0, 255, 0]]
        colored_image = np.zeros_like(image)
        for i, threshold in enumerate(thresholds[:-1]):
            mask = (normalized >= threshold) & (normalized < thresholds[i + 1])
            colored_image[mask] = colors[i]
        return colored_image
