from gradio_gui import GradioGui
from segmentation_processor import WatershedProcessor

STANDARD_WIDTH = 512
STANDARD_HEIGHT = 512

if __name__ == "__main__":
    processor = WatershedProcessor(STANDARD_WIDTH, STANDARD_HEIGHT)
    gui = GradioGui(STANDARD_WIDTH, STANDARD_HEIGHT)
    gui.run(processor)