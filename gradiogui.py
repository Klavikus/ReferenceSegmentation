import gradio as gr

from segmentation_processor import WatershedProcessor


class GradioGui:
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

    def run(self, processor: WatershedProcessor, share: bool = False):
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
                    result_image = gr.Image(
                        type="numpy",
                        label="Результат",
                        elem_id="result_image",
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
                fn=processor.prepare_image,
                inputs=input_image,
                outputs=[output_image, mask_input_f, mask_input_b]
            )

            run_btn.click(
                fn=processor.watershed_algorithm,
                inputs=[output_image, mask_input_f, mask_input_b],
                outputs=[result_image, data_table, download_excel, data_table_overall, download_excel_overall]
            )

            def auto_run_processing(auto_update_flag, output_img, mask_f, mask_b):
                if auto_update_flag:
                    return processor.watershed_algorithm(output_img, mask_f, mask_b)

            mask_input_f.change(
                fn=auto_run_processing,
                inputs=[auto_update, output_image, mask_input_f, mask_input_b],
                outputs=[result_image, data_table, download_excel, data_table_overall, download_excel_overall]
            )

            mask_input_b.change(
                fn=auto_run_processing,
                inputs=[auto_update, output_image, mask_input_f, mask_input_b],
                outputs=[result_image, data_table, download_excel, data_table_overall, download_excel_overall]
            )

            demo.launch(share=share, debug=True)
