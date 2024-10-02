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

    def preprocess_image(self, image: np.ndarray) -> Any | None:
        if image is None:
            return None, 0, 0
        h, w = image.shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        x_offset = (self.width - new_w) // 2
        y_offset = (self.height - new_h) // 2

        if len(image.shape) == 2:
            padded = np.full((self.height, self.width), 255, dtype=image.dtype)
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        else:
            padded = np.full((self.height, self.width, 3), 255, dtype=image.dtype)
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized

        return padded, x_offset, y_offset

    def prepare_image(self, image: np.ndarray) -> tuple:
        if image is None:
            return None, None, None, None, None
        resized_image, _, _ = self.preprocess_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)
        return gray_rgb, gray_rgb, gray_rgb

    def watershed_algorithm(self, image: np.ndarray, mask_data_f: dict, mask_data_b: dict) -> tuple:
        if image is None or mask_data_f is None or mask_data_b is None:
            return "Пожалуйста, сначала загрузите изображение и нарисуйте маски.", None, None, None, None

        if 'layers' not in mask_data_f or not mask_data_f['layers']:
            return "Пожалуйста, нарисуйте маску в Mask F.", None, None, None, None
        if 'layers' not in mask_data_b or not mask_data_b['layers']:
            return "Пожалуйста, нарисуйте маску в Mask B.", None, None, None, None

        mask_sf = mask_data_f['layers'][0]
        if mask_sf is None:
            return "Пожалуйста, нарисуйте маску в Mask F.", None, None, None, None

        mask_sb = mask_data_b['layers'][0]
        if mask_sb is None:
            return "Пожалуйста, нарисуйте маску в Mask B.", None, None, None, None

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mask_sf = np.array(mask_sf)
        mask_sf = cv2.cvtColor(mask_sf, cv2.COLOR_RGBA2GRAY)

        mask_sb = np.array(mask_sb)
        mask_sb = cv2.cvtColor(mask_sb, cv2.COLOR_RGBA2GRAY)

        mask_sf, x_offset, y_offset = self.preprocess_image(mask_sf)
        mask_sb, _, _ = self.preprocess_image(mask_sb)

        _, sure_fg = cv2.threshold(mask_sf, 0, 255, cv2.THRESH_BINARY)
        _, sure_bg = cv2.threshold(mask_sb, 0, 255, cv2.THRESH_BINARY)

        markers = np.zeros_like(sure_fg, dtype=np.int32)
        markers[sure_bg > 0] = 1
        markers[sure_fg > 0] = 255

        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        markers = cv2.watershed(image_bgr, markers)
        result = image_bgr.copy()

        unique_markers = np.unique(markers)
        unique_markers = unique_markers[(unique_markers != 1) & (unique_markers != -1)]

        data = []
        current_object_id = 1

        for marker in unique_markers:
            obj_mask = np.uint8(markers == marker)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)

            for label in range(1, num_labels):  # label=0 это фон внутри маски
                component_mask = np.uint8(labels == label)
                mask = labels == label

                contours, _ = cv2.findContours(component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]
                    area = np.sum(component_mask) * self.pixel_to_nm
                    perimeter = cv2.arcLength(contour, True)

                    m = cv2.moments(contour)
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

                    object_image = np.zeros_like(image_bgr)
                    object_image[mask] = image_bgr[mask]

                    colored_object = self.assigne_color_with_normilize(object_image, mask)
                    alpha = 0.35
                    result[mask] = cv2.addWeighted(result[mask], 1 - alpha, colored_object[mask], alpha, 0)

                    cv2.drawContours(result, [contour], -1, (0, 0, 255), 1)
                    cv2.circle(result, (c_x, c_y), 3, (0, 255, 0), -1)

                    current_object_id += 1

        if not data:
            return "Объекты не обнаружены.", None, None, None, None

        df = pd.DataFrame(data)

        excel_path = "output.xlsx"
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Сегментированные объекты')

        if not Path(excel_path).is_file():
            return "Ошибка при сохранении Excel файла.", None, None, None, None

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

        with pd.ExcelWriter(excel_path_overall, engine='xlsxwriter') as writer:
            df_overall.to_excel(writer, index=False, sheet_name='Общие данные')

        if not Path(excel_path_overall).is_file():
            return "Ошибка при сохранении общего Excel файла.", None, None, None, None

        return result, df, excel_path, df_overall, excel_path_overall

    def assigne_color_with_normilize(self, image: np.ndarray, mask: bool) -> np.ndarray | None:
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        min_val = np.min(gray[mask])
        max_val = np.max(gray[mask])
        normalized = np.clip((gray - min_val) / (max_val - min_val), 0, 1)

        n = 4
        thresholds = np.linspace(0, 1, n + 1)
        colors = [
            [255, 0, 0],  # Red
            [255, 165, 0],  # Orange
            [255, 255, 0],  # Yellow
            [0, 255, 0],  # Green
        ]

        result = np.zeros_like(image)

        for i in range(n):
            stage_mask = (normalized >= thresholds[i]) & (normalized < thresholds[i + 1])
            result[stage_mask] = colors[i]

        return result
