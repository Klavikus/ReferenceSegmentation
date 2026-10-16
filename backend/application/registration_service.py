from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from backend.contracts.registration_models import (
    RegistrationConfig,
    RegistrationRun,
)
from backend.core.registration import register_image_pair
from backend.infrastructure.exporters.registration_exporter import (
    write_registration_export,
)


class RegistrationService:
    def __init__(self, config: RegistrationConfig | None = None):
        self.config = config or RegistrationConfig()

    def align_pair(
        self,
        before: np.ndarray,
        after: np.ndarray,
        bottom_exclusion_px: int | None = None,
    ) -> RegistrationRun:
        config = self.config
        if bottom_exclusion_px is not None:
            config = replace(
                config,
                bottom_exclusion_px=max(0, int(round(bottom_exclusion_px))),
            )

        result = register_image_pair(before, after, config)
        output_dir = Path(tempfile.mkdtemp(prefix="registration-"))
        export = write_registration_export(result, output_dir)
        return RegistrationRun(result=result, export=export)
