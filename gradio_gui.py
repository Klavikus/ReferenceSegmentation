import math

import gradio as gr
import numpy as np
from backend.application.pipeline_service import (
    MARKER_MODE_CONTOUR,
    MARKER_MODE_EMPTY,
    MARKER_MODE_FULL,
    PipelineService,
)
from backend.application.registration_service import RegistrationService
from backend.core.ruler import measure_ruler_distance
from backend.contracts.pipeline_models import PreparedSample, SampleSegmentationRun


def build_launch_kwargs(
    css: str,
    host: str,
    port: int,
    debug: bool,
    share: bool,
) -> dict[str, str | int | bool]:
    return {
        "css": css,
        "server_name": host,
        "server_port": port,
        "debug": debug,
        "share": share,
    }


class GradioGui:
    def __init__(self, width: int = 512, height: int = 512, sam2_generator=None):
        self.processor = None
        self.width = width
        self.height = height
        self._prepared_sample: PreparedSample | None = None
        self._latest_run: SampleSegmentationRun | None = None
        self._last_marker_suggestion = None
        self._last_marker_source = "manual"
        self.registration_service: RegistrationService | None = None

    def run(
        self,
        processor: PipelineService,
        host: str = "127.0.0.1",
        port: int = 7860,
        debug: bool = False,
        share: bool = False,
        registration_service: RegistrationService | None = None,
    ):
        self.processor = processor
        self.registration_service = registration_service
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
        with gr.Blocks() as demo:
            with gr.Tabs():
                # Вкладка 1: Сегментация
                with gr.TabItem("Сегментация"):
                    self.build_segmentation_tab()
                # Вкладка 2: Линейка
                with gr.TabItem("Линейка"):
                    self.build_ruler_tab()
                if self.registration_service is not None:
                    with gr.TabItem("Совмещение"):
                        self.build_registration_tab()
            launch_kwargs = build_launch_kwargs(
                css=css,
                host=host,
                port=port,
                debug=debug,
                share=share,
            )
            demo.launch(**launch_kwargs)

    # ========================== Сегментация ==================================
    def build_segmentation_tab(self):
        with gr.Row(visible=True):
            input_column = self.build_input_column()
            output_column = self.build_output_column()
            controls_column, extra_components = self.build_controls_column()
        self.setup_segmentation_event_listeners(
            input_column, output_column, extra_components
        )

    def build_input_column(self):
        with gr.Column(
            scale=1, min_width=200, elem_classes="gradio-column"
        ) as input_column:
            input_image = gr.Image(
                type="numpy",
                label="Входное изображение",
                height=self.height,
                width=self.width,
            )
            mask_input_f = gr.ImageEditor(
                label="Метки объектов",
                brush=gr.Brush(colors=["#0000EE"]),
                height=self.height,
                width=self.width,
                canvas_size=(self.width, self.height),
                fixed_canvas=True,
                interactive=True,
            )
        input_column.input_image = input_image
        input_column.mask_input_f = mask_input_f
        return input_column

    def build_output_column(self):
        with gr.Column(
            scale=1, min_width=200, elem_classes="gradio-column"
        ) as output_column:
            output_image = gr.Image(
                type="numpy",
                label="Серое изображение",
                height=self.height,
                width=self.width,
            )
            mask_input_b = gr.ImageEditor(
                label="Метки фона",
                brush=gr.Brush(colors=["#EE0000"]),
                height=self.height,
                width=self.width,
                canvas_size=(self.width, self.height),
                fixed_canvas=True,
                interactive=True,
            )
        output_column.output_image = output_image
        output_column.mask_input_b = mask_input_b
        return output_column

    def build_controls_column(self):
        with gr.Column(
            scale=1, min_width=200, elem_classes="gradio-column"
        ) as controls_column:
            with gr.Row():
                prepare_btn = gr.Button("Подготовить")
                if self.processor and self.processor.can_suggest_markers():
                    suggest_markers_btn = gr.Button("Предложить маркеры")
                else:
                    suggest_markers_btn = gr.Button(
                        "Предложить маркеры",
                        interactive=False,
                    )
                object_marker_mode = gr.Dropdown(
                    choices=[MARKER_MODE_FULL, MARKER_MODE_CONTOUR, MARKER_MODE_EMPTY],
                    value=MARKER_MODE_FULL,
                    label="Метки объектов",
                )
                background_marker_mode = gr.Dropdown(
                    choices=[MARKER_MODE_FULL, MARKER_MODE_CONTOUR, MARKER_MODE_EMPTY],
                    value=MARKER_MODE_FULL,
                    label="Метки фона",
                )
                marker_contour_thickness = gr.Number(
                    label="Толщина контура",
                    value=3,
                    step=1,
                    minimum=1,
                    maximum=100,
                )
                apply_marker_modes_btn = gr.Button("Применить режимы")
                auto_update = gr.Checkbox(
                    label="Автоматически обновлять",
                    value=True,
                )
                run_btn = gr.Button("Рассчитать")
                pixel_to_nm = gr.Number(
                    label="pixel_to_nm",
                    value=1.0,
                    precision=4,
                )
                contour_thickness = gr.Number(
                    label="contour_thickness",
                    value=3.0,
                    step=1,
                    minimum=1,
                    maximum=100,
                )
            with gr.Tabs():
                with gr.TabItem("Контуры"):
                    result_image_contours = gr.Image(
                        type="numpy",
                        label="Результат (Контуры)",
                        height=self.height,
                        width=self.width,
                    )
                with gr.TabItem("Покрас по нормали"):
                    result_image_normal = gr.Image(
                        type="numpy",
                        label="Результат (Покрас по нормали)",
                        height=self.height,
                        width=self.width,
                    )
                with gr.TabItem("Покрас с персентилями"):
                    result_image_percentile = gr.Image(
                        type="numpy",
                        label="Результат (Покрас с персентилями)",
                        height=self.height,
                        width=self.width,
                    )
            data_table = gr.Dataframe(
                label="Данные сегментированных объектов",
                headers=["ID", "Площадь", "Периметр", "Центр X", "Центр Y"],
                interactive=False,
            )
            data_table_overall = gr.Dataframe(label="Общие данные", interactive=False)
            with gr.Row():
                download_excel = gr.File(label="Скачать Excel Файл")
                download_excel_overall = gr.File(label="Скачать Excel Файл")
                download_contours_btn = gr.Button("Скачать контуры по слоям")
                download_contours_file = gr.File(label="Скачать ZIP архив контуров")
        extra = {
            "prepare_btn": prepare_btn,
            "suggest_markers_btn": suggest_markers_btn,
            "object_marker_mode": object_marker_mode,
            "background_marker_mode": background_marker_mode,
            "marker_contour_thickness": marker_contour_thickness,
            "apply_marker_modes_btn": apply_marker_modes_btn,
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
            "download_contours_file": download_contours_file,
        }
        return controls_column, extra

    def setup_segmentation_event_listeners(self, input_column, output_column, extra):
        input_image = input_column.input_image
        mask_input_f = input_column.mask_input_f
        output_image = output_column.output_image
        mask_input_b = output_column.mask_input_b

        auto_update = extra["auto_update"]
        prepare_btn = extra["prepare_btn"]
        suggest_markers_btn = extra["suggest_markers_btn"]
        object_marker_mode = extra["object_marker_mode"]
        background_marker_mode = extra["background_marker_mode"]
        marker_contour_thickness = extra["marker_contour_thickness"]
        apply_marker_modes_btn = extra["apply_marker_modes_btn"]
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
            outputs=[output_image, mask_input_f, mask_input_b],
        )
        suggest_markers_btn.click(
            fn=self.__handle_suggest_markers,
            inputs=[input_image, output_image],
            outputs=[mask_input_f, mask_input_b],
        )
        apply_marker_modes_btn.click(
            fn=self.__handle_apply_marker_modes,
            inputs=[
                object_marker_mode,
                background_marker_mode,
                marker_contour_thickness,
            ],
            outputs=[mask_input_f, mask_input_b],
        )
        run_btn.click(
            fn=self.__handle_watershed_algorithm,
            inputs=[output_image, mask_input_f, mask_input_b, pixel_to_nm],
            outputs=[
                result_image_contours,
                result_image_percentile,
                result_image_normal,
                data_table,
                download_excel,
                data_table_overall,
                download_excel_overall,
            ],
        )
        mask_input_f.change(
            fn=self.__auto_run_processing,
            inputs=[auto_update, output_image, mask_input_f, mask_input_b, pixel_to_nm],
            outputs=[
                result_image_contours,
                result_image_percentile,
                result_image_normal,
                data_table,
                download_excel,
                data_table_overall,
                download_excel_overall,
            ],
        )
        mask_input_b.change(
            fn=self.__auto_run_processing,
            inputs=[auto_update, output_image, mask_input_f, mask_input_b, pixel_to_nm],
            outputs=[
                result_image_contours,
                result_image_percentile,
                result_image_normal,
                data_table,
                download_excel,
                data_table_overall,
                download_excel_overall,
            ],
        )
        download_contours_btn.click(
            fn=self.__handle_download_contours,
            inputs=contour_thickness,
            outputs=download_contours_file,
        )

    # ========================== Совмещение пары ==============================
    def build_registration_tab(self):
        gr.Markdown(
            "Автоматическое совмещение изображений «до / после»: только поворот "
            "и перенос, без изменения масштаба. Результат обрезается до общей области."
        )
        with gr.Row():
            with gr.Column():
                registration_before = gr.Image(
                    type="numpy", label="До", height=self.height,
                )
            with gr.Column():
                registration_after = gr.Image(
                    type="numpy", label="После", height=self.height,
                )

        gr.Markdown(
            "Нижнюю служебную область микроскопа (легенда, подписи, масштабная "
            "линейка) можно исключить из поиска соответствий. Она также не попадёт "
            "в итоговую общую область."
        )
        registration_bottom_exclusion = gr.Number(
            label="Исключить снизу, px",
            value=self.registration_service.config.bottom_exclusion_px,
            precision=0,
            interactive=True,
        )
        registration_run_btn = gr.Button("Совместить изображения", variant="primary")
        with gr.Row():
            with gr.Column():
                registration_before_output = gr.Image(
                    type="numpy", label="До — общая область",
                    height=self.height, interactive=False,
                )
            with gr.Column():
                registration_after_output = gr.Image(
                    type="numpy", label="После — совмещено",
                    height=self.height, interactive=False,
                )
        registration_overlay = gr.Image(
            type="numpy", label="Контроль совмещения (overlay)",
            height=self.height, interactive=False,
        )
        registration_diagnostics = gr.JSON(label="Диагностика совмещения")
        with gr.Row():
            registration_before_file = gr.File(label="Скачать До — aligned")
            registration_after_file = gr.File(label="Скачать После — aligned")
            registration_zip_file = gr.File(label="Скачать ZIP")
        registration_run_btn.click(
            fn=self.__handle_registration,
            inputs=[
                registration_before,
                registration_after,
                registration_bottom_exclusion,
            ],
            outputs=[
                registration_before_output,
                registration_after_output,
                registration_overlay,
                registration_diagnostics,
                registration_before_file,
                registration_after_file,
                registration_zip_file,
            ],
        )

    def __handle_registration(self, before_image, after_image, bottom_exclusion_px):
        if self.registration_service is None:
            raise gr.Error("Сервис совмещения не настроен")
        if before_image is None or after_image is None:
            raise gr.Error("Загрузите оба изображения: до и после")
        try:
            run = self.registration_service.align_pair(
                np.asarray(before_image),
                np.asarray(after_image),
                bottom_exclusion_px=(
                    0 if bottom_exclusion_px is None else int(bottom_exclusion_px)
                ),
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise gr.Error(f"Ошибка совмещения изображений: {e}")

        diagnostics = run.result.diagnostics.to_dict()
        diagnostics["transform_after_to_before"] = run.result.transform_matrix.tolist()
        diagnostics["crop_rect"] = {
            "x": run.result.crop_rect.x,
            "y": run.result.crop_rect.y,
            "width": run.result.crop_rect.width,
            "height": run.result.crop_rect.height,
        }
        return (
            run.result.before_aligned,
            run.result.after_aligned,
            run.result.overlay_preview,
            diagnostics,
            str(run.export.before_path),
            str(run.export.after_path),
            str(run.export.zip_path),
        )

    # ========================== Функционал Линейки ===========================
    def build_ruler_tab(self):
        with gr.Row():
            with gr.Column() as ruler_column_input:
                gr.Markdown(
                    "Нарисуйте горизонтальную линию на изображении для измерения расстояния"
                )
                ruler_input_image = gr.ImageEditor(
                    label="Изображение для измерения",
                    height=self.height,
                    width=self.width,
                    brush=gr.Brush(colors=["#FF0000"]),
                )
                ruler_size_nm = gr.Number(
                    label="Размер линейки (нм)",
                    value=0.0,
                    precision=4,
                    interactive=True,
                )
                ruler_measure_btn = gr.Button("Измерить")
            with gr.Column() as ruler_column_output:
                ruler_distance_output = gr.Number(label="Расстояние (пиксели)")
                ruler_px_to_nm_output = gr.Number(
                    label="px_to_nm", value=0.0, precision=4, interactive=False
                )
        ruler_measure_btn.click(
            fn=self.__handle_measure_distance,
            inputs=[ruler_input_image, ruler_size_nm],
            outputs=[ruler_distance_output, ruler_px_to_nm_output],
        )
        return ruler_column_input, ruler_column_output

    def __handle_measure_distance(self, image, ruler_size_nm):
        try:
            img_array = np.array(image["layers"][0])
            return measure_ruler_distance(img_array, ruler_size_nm)
        except Exception as e:
            return str(e), 0.0

    # ========================== Методы для сегментации =======================
    def __handle_prepare_image(self, input_image):
        prepared_sample = self.processor.prepare_sample(input_image)
        self._prepared_sample = prepared_sample
        self._latest_run = None
        self._last_marker_suggestion = None
        self._last_marker_source = "manual"
        output_image = prepared_sample.grayscale_preview
        return output_image, output_image, output_image

    def __handle_suggest_markers(self, input_image, prepared_image):
        if input_image is None:
            raise gr.Error("Сначала подготовьте изображение")
        if self._prepared_sample is None:
            raise gr.Error("Сначала подготовьте изображение")
        if not self.processor.can_suggest_markers():
            raise gr.Error("SAM2 недоступна. Проверьте установку и наличие GPU.")
        self._last_marker_suggestion = None
        self._last_marker_source = "manual"
        try:
            suggestion = self.processor.suggest_markers(
                sample_id=self._prepared_sample.sample_id,
                image=prepared_image,
            )
            self._last_marker_suggestion = suggestion
            self._last_marker_source = "sam2"
            return suggestion.foreground_payload, suggestion.background_payload
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise gr.Error(f"Ошибка генерации маркеров: {e}")

    def __handle_apply_marker_modes(
        self, object_marker_mode, background_marker_mode, marker_contour_thickness
    ):
        if self._prepared_sample is None:
            raise gr.Error("Сначала подготовьте изображение")
        if self._last_marker_suggestion is None:
            raise gr.Error("Сначала предложите маркеры через SAM2")
        thickness_value = float(marker_contour_thickness)
        if not thickness_value.is_integer():
            raise gr.Error("Толщина контура должна быть целым числом")
        contour_thickness = int(thickness_value)
        try:
            return self.processor.create_marker_payloads(
                sample_id=self._prepared_sample.sample_id,
                foreground_mask=self._last_marker_suggestion.foreground_mask,
                background_mask=self._last_marker_suggestion.background_mask,
                foreground_mode=object_marker_mode,
                background_mode=background_marker_mode,
                contour_thickness=contour_thickness,
            )
        except Exception as e:
            raise gr.Error(f"Ошибка применения режимов маркеров: {e}")

    def __handle_watershed_algorithm(
        self, output_image, mask_input_f, mask_input_b, pixel_to_nm
    ):
        if self._prepared_sample is None:
            raise gr.Error("Сначала подготовьте изображение")

        run = self.processor.segment_sample(
            sample_id=self._prepared_sample.sample_id,
            foreground_payload=mask_input_f,
            background_payload=mask_input_b,
            pixel_to_nm=pixel_to_nm,
            marker_source=self._last_marker_source,
        )
        self._latest_run = run
        return (
            run.segmentation.contour_preview,
            run.segmentation.percentile_preview,
            run.segmentation.normalized_preview,
            run.export.object_table,
            run.export.object_table_path,
            run.export.overall_table,
            run.export.overall_table_path,
        )

    def __auto_run_processing(
        self, auto_update_flag, output_img, mask_f, mask_b, pixel_to_nm
    ):
        if auto_update_flag:
            return self.__handle_watershed_algorithm(
                output_img, mask_f, mask_b, pixel_to_nm
            )
        return (None, None, None, None, None, None, None)

    def __handle_download_contours(self, contour_thickness):
        if self._latest_run is None or len(self._latest_run.segmentation.contours_original) == 0:
            raise gr.Error(
                "Нет данных контуров для скачивания. Сначала запустите обработку."
            )
        if self._prepared_sample is None:
            raise gr.Error("Нет подготовленного изображения для экспорта контуров.")
        result = self.processor.export_contours_zip(
            sample_id=self._prepared_sample.sample_id,
            contours_original=self._latest_run.segmentation.contours_original,
            thickness=int(contour_thickness),
        )
        return result.zip_path
