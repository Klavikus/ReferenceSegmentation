import gradio as gr

from segmentation_processor import WatershedProcessor


class GradioGui:
    def __init__(self, width: int = 512, height: int = 512):
        self.processor = None
        self.width = width
        self.height = height

    def run(self, processor: WatershedProcessor, share: bool = False):
        self.processor = processor
        with gr.Blocks(css="""
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            .gradio-row {
                flex-wrap: wrap;
            }
            .gradio-column {
                # display: flex;
                # flex-direction: column;
                align-items: center;
            }
            """) as demo:
            with gr.Row(elem_id="images_row", visible=True):
                with gr.Column(scale=1, min_width=200, elem_classes="gradio-column"):
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
                        brush=gr.Brush(colors=['#0000EE']),
                        height=self.height,
                        width=self.width,
                        interactive=True,
                        elem_id="mask_inputF"
                    )
                with gr.Column(scale=1, min_width=200, elem_classes="gradio-column"):
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
                        brush=gr.Brush(colors=['#EE0000']),
                        height=self.height,
                        width=self.width,
                        interactive=True,
                        elem_id="mask_inputB"
                    )

                with gr.Column(scale=1, min_width=200, elem_classes="gradio-column"):
                    with gr.Row():
                        prepare_btn = gr.Button("Подготовить", elem_id="save_btn")
                        auto_update = gr.Checkbox(
                            label="Автоматически обновлять",
                            value=True,
                            elem_id="auto_update")
                        run_btn = gr.Button("Рассчитать", elem_id="run_btn")

                    with gr.Tabs():
                        with gr.TabItem("Контуры"):
                            result_image_contours = gr.Image(
                                type="numpy",
                                label="Результат (Контуры)",
                                elem_id="result_image_contours",
                                height=self.height,
                                width=self.width)
                        with gr.TabItem("Покрас по нормали"):
                            result_image_normal = gr.Image(
                                type="numpy",
                                label="Результат (Покрас по нормали)",
                                elem_id="result_image_normal",
                                height=self.height,
                                width=self.width)
                        with gr.TabItem("Покрас с персентилями"):
                            result_image_percentile = gr.Image(
                                type="numpy",
                                label="Результат (Покрас с персентилями)",
                                elem_id="result_image_percentile",
                                height=self.height,
                                width=self.width)

                    data_table = gr.Dataframe(
                        label="Данные сегментированных объектов",
                        headers=["ID", "Площадь", "Периметр", "Центр X", "Центр Y"],
                        interactive=False,
                    )
                    data_table_overall = gr.Dataframe(
                        label="Общие данные",
                        interactive=False,
                    )
                    with gr.Row():
                        download_excel = gr.File(label="Скачать Excel Файл")
                        download_excel_overall = gr.File(label="Скачать Excel Файл")

            prepare_btn.click(
                fn=self.__handle_prepare_image,
                inputs=input_image,
                outputs=[output_image, mask_input_f, mask_input_b]
            )

            run_btn.click(
                fn=self.__handle_watershed_algorithm,
                inputs=[output_image, mask_input_f, mask_input_b],
                outputs=[result_image_contours, result_image_percentile, result_image_normal, data_table,
                         download_excel, data_table_overall, download_excel_overall]
            )

            mask_input_f.change(
                fn=self.__auto_run_processing,
                inputs=[auto_update, output_image, mask_input_f, mask_input_b],
                outputs=[result_image_contours, result_image_percentile, result_image_normal, data_table,
                         download_excel, data_table_overall, download_excel_overall]
            )

            mask_input_b.change(
                fn=self.__auto_run_processing,
                inputs=[auto_update, output_image, mask_input_f, mask_input_b],
                outputs=[result_image_contours, result_image_percentile, result_image_normal, data_table,
                         download_excel, data_table_overall, download_excel_overall]
            )

            demo.launch(share=share, debug=True)

    def __handle_prepare_image(self, input_image):
        output_image = self.processor.prepare_image(input_image)
        return output_image, output_image, output_image

    def __handle_watershed_algorithm(self, output_image, mask_input_f, mask_input_b):
        return self.processor.watershed_algorithm(output_image, mask_input_f, mask_input_b)

    def __auto_run_processing(self, auto_update_flag, output_img, mask_f, mask_b):
        if auto_update_flag:
            return self.__handle_watershed_algorithm(output_img, mask_f, mask_b)
