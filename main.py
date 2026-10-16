from backend.application.pipeline_service import PipelineService
from backend.application.registration_service import RegistrationService
from backend.contracts.pipeline_models import PipelineConfig
from backend.infrastructure.markers import Sam2MarkerAdapter
from gradio_gui import GradioGui
from runtime_config import load_runtime_config

STANDARD_WIDTH = 512
STANDARD_HEIGHT = 512

if __name__ == "__main__":
    runtime_config = load_runtime_config()
    pipeline_service = PipelineService(
        config=PipelineConfig.standard(),
        sam2_adapter=Sam2MarkerAdapter(),
    )
    gui = GradioGui(STANDARD_WIDTH, STANDARD_HEIGHT)
    gui.run(
        pipeline_service,
        host=runtime_config.host,
        port=runtime_config.port,
        debug=runtime_config.debug,
        share=runtime_config.share,
        registration_service=RegistrationService(),
    )
