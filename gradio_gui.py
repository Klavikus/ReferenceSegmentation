import math
import os
import tempfile
import zipfile
import gradio as gr
import numpy as np
from PIL import Image
from segmentation_processor import WatershedProcessor


class GradioGui:
    def __init__(self, width: int = 512, height: int = 512):
        self.processor = None
        self.width = width
        self.height = height

    def run(self, processor: WatershedProcessor, share: bool = False):
        self.processor = processor
        css = """
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            .gradio-row {
                flex-wrap: wrap;
            }
            .gradio-column {
                align-items: center;
            }
        """
        with gr.Blocks(css=css) as demo:
            with gr.Tabs():
                # Вкладка 1: Сегментация
                with gr.TabItem("Сегментация"):
                    self.build_segmentation_tab()
                # Вкладка 2: Линейка
                with gr.TabItem("Линейка"):
                    self.build_ruler_tab()
            demo.launch(share=share, debug=True)

    # ========================== Сегментация ==================================
    def build_segmentation_tab(self):
        with gr.Row(elem_id="images_row", visible=True):
            input_column = self.build_input_column()
            output_column = self.build_output_column()
            controls_column, extra_components = self.build_controls_column()
        self.setup_segmentation_event_listeners(input_column, output_column, extra_components)

    def build_input_column(self):
        with gr.Column(scale=1, min_width=200, elem_classes="gradio-column") as input_column:
            input_image = gr.Image(
                type="numpy",
                label="Входное изображение",
                elem_id="input_image",
                height=self.height,
                width=self.width
            )
            mask_input_f = gr.ImageEditor(
                label="Метки объектов",
                show_fullscreen_button=True,
                brush=gr.Brush(colors=["#0000EE"]),
                height=self.height,
                width=self.width,
                interactive=True,
                elem_id="mask_inputF"
            )
        input_column.input_image = input_image
        input_column.mask_input_f = mask_input_f
        return input_column

    def build_output_column(self):
        with gr.Column(scale=1, min_width=200, elem_classes="gradio-column") as output_column:
            output_image = gr.Image(
                type="numpy",
                label="Серое изображение",
                elem_id="output_image",
                height=self.height,
                width=self.width
            )
            mask_input_b = gr.ImageEditor(
                label="Метки фона",
                show_fullscreen_button=True,
                brush=gr.Brush(colors=["#EE0000"]),
                height=self.height,
                width=self.width,
                interactive=True,
                elem_id="mask_inputB"
            )
        output_column.output_image = output_image
        output_column.mask_input_b = mask_input_b
        return output_column

    def build_controls_column(self):
        with gr.Column(scale=1, min_width=200, elem_classes="gradio-column") as controls_column:
            with gr.Row():
                prepare_btn = gr.Button("Подготовить", elem_id="save_btn")
                auto_update = gr.Checkbox(
                    label="Автоматически обновлять",
                    value=True,
                    elem_id="auto_update"
                )
                run_btn = gr.Button("Рассчитать", elem_id="run_btn")
                pixel_to_nm = gr.Number(
                    label="pixel_to_nm",
                    value=1.0,
                    precision=4,
                    elem_id="pixel_to_nm"
                )
                contour_thickness = gr.Number(
                    label="contour_thickness",
                    value=3.0,
                    step=1,
                    minimum=1,
                    maximum=100,
                    elem_id="contour_thickness"
                )
            with gr.Tabs():
                with gr.TabItem("Контуры"):
                    result_image_contours = gr.Image(
                        type="numpy",
                        label="Результат (Контуры)",
                        elem_id="result_image_contours",
                        height=self.height,
                        width=self.width
                    )
                with gr.TabItem("Покрас по нормали"):
                    result_image_normal = gr.Image(
                        type="numpy",
                        label="Результат (Покрас по нормали)",
                        elem_id="result_image_normal",
                        height=self.height,
                        width=self.width
                    )
                with gr.TabItem("Покрас с персентилями"):
                    result_image_percentile = gr.Image(
                        type="numpy",
                        label="Результат (Покрас с персентилями)",
                        elem_id="result_image_percentile",
                        height=self.height,
                        width=self.width
                    )
            data_table = gr.Dataframe(
                label="Данные сегментированных объектов",
                headers=["ID", "Площадь", "Периметр", "Центр X", "Центр Y"],
                interactive=False
            )
            data_table_overall = gr.Dataframe(
                label="Общие данные",
                interactive=False
            )
            with gr.Row():
                download_excel = gr.File(label="Скачать Excel Файл")
                download_excel_overall = gr.File(label="Скачать Excel Файл")
                download_contours_btn = gr.Button("Скачать контуры по слоям")
                download_contours_file = gr.File(label="Скачать ZIP архив контуров")
        extra = {
            "prepare_btn": prepare_btn,
            "auto_update": auto_update,
            "run_btn": run_btn,
            "pixel_to_nm": pixel_to_nm,
            "contour_thickness": contour_thickness,
            "result_image_contours": result_image_contours,
            "result_image_normal": result_image_normal,
            "result_image_percentile": result_image_percentile,
            "data_table": data_table,
            "download_excel": download_excel,
            "data_table_overall": data_table_overall,
            "download_excel_overall": download_excel_overall,
            "download_contours_btn": download_contours_btn,
            "download_contours_file": download_contours_file
        }
        return controls_column, extra

    def setup_segmentation_event_listeners(self, input_column, output_column, extra):
        input_image = input_column.input_image
        mask_input_f = input_column.mask_input_f
        output_image = output_column.output_image
        mask_input_b = output_column.mask_input_b

        auto_update = extra["auto_update"]
        prepare_btn = extra["prepare_btn"]
        run_btn = extra["run_btn"]
        pixel_to_nm = extra["pixel_to_nm"]
        contour_thickness = extra["contour_thickness"]

        result_image_contours = extra["result_image_contours"]
        result_image_normal = extra["result_image_normal"]
        result_image_percentile = extra["result_image_percentile"]
        data_table = extra["data_table"]
        download_excel = extra["download_excel"]
        data_table_overall = extra["data_table_overall"]
        download_excel_overall = extra["download_excel_overall"]
        download_contours_btn = extra["download_contours_btn"]
        download_contours_file = extra["download_contours_file"]

        # Привязка событий для сегментации
        prepare_btn.click(
            fn=self.__handle_prepare_image,
            inputs=input_image,
            outputs=[output_image, mask_input_f, mask_input_b]
        )
        run_btn.click(
            fn=self.__handle_watershed_algorithm,
            inputs=[output_image, mask_input_f, mask_input_b, pixel_to_nm],
            outputs=[result_image_contours, result_image_percentile, result_image_normal,
                     data_table, download_excel, data_table_overall, download_excel_overall]
        )
        mask_input_f.change(
            fn=self.__auto_run_processing,
            inputs=[auto_update, output_image, mask_input_f, mask_input_b, pixel_to_nm],
            outputs=[result_image_contours, result_image_percentile, result_image_normal,
                     data_table, download_excel, data_table_overall, download_excel_overall]
        )
        mask_input_b.change(
            fn=self.__auto_run_processing,
            inputs=[auto_update, output_image, mask_input_f, mask_input_b, pixel_to_nm],
            outputs=[result_image_contours, result_image_percentile, result_image_normal,
                     data_table, download_excel, data_table_overall, download_excel_overall]
        )
        download_contours_btn.click(
            fn=self.__handle_download_contours,
            inputs=contour_thickness,
            outputs=download_contours_file
        )

    # ========================== Функционал Линейки ===========================
    def build_ruler_tab(self):
        with gr.Column(elem_id="ruler_column", scale=1) as ruler_column:
            ruler_instructions = gr.Markdown("Нарисуйте горизонтальную линию на изображении для измерения расстояния.")
            ruler_input_image = gr.ImageEditor(
                label="Изображение для измерения",
                height=self.height,
                width=self.width,
                brush=gr.Brush(colors=["#FF0000"]),
                show_fullscreen_button=True,
            )
            ruler_measure_btn = gr.Button("Измерить")
            ruler_distance_output = gr.Number(label="Расстояние (пиксели)")

        # Привязка кнопки "Измерить"
        ruler_measure_btn.click(
            fn=self.__handle_measure_distance,
            inputs=ruler_input_image,
            outputs=ruler_distance_output
        )

    def __handle_measure_distance(self, image):
        try:
            img_array = np.array(image['layers'][0])

            red_pixels = img_array[:, :, 0] > 100  # Маска для красных пикселей

            red_pixel_coords = np.where(red_pixels)

            min_x = np.min(red_pixel_coords[1])
            max_x = np.max(red_pixel_coords[1])

            distance = max_x - min_x

            return distance
        except Exception as e:
            return str(e)

    # ========================== Методы для сегментации =======================
    def __handle_prepare_image(self, input_image):
        output_image = self.processor.prepare_image(input_image)
        return output_image, output_image, output_image

    def __handle_watershed_algorithm(self, output_image, mask_input_f, mask_input_b, pixel_to_nm):
        return self.processor.watershed_algorithm(output_image, mask_input_f, mask_input_b, pixel_to_nm)

    def __auto_run_processing(self, auto_update_flag, output_img, mask_f, mask_b, pixel_to_nm):
        if auto_update_flag:
            return self.__handle_watershed_algorithm(output_img, mask_f, mask_b, pixel_to_nm)

    def __handle_download_contours(self, contour_thickness):
        if not hasattr(self.processor, 'contours_original') or len(self.processor.contours_original) == 0:
            raise gr.Error("Нет данных контуров для скачивания. Сначала запустите обработку.")
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "contours.zip")
        contour_images = self.processor.generate_contour_images(int(contour_thickness))
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename, img_array in contour_images:
                pil_img = Image.fromarray(img_array, "RGBA")
                temp_img_path = os.path.join(temp_dir, filename)
                pil_img.save(temp_img_path)
                zipf.write(temp_img_path, arcname=filename)
        return zip_path
