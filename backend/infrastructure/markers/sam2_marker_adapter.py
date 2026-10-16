from typing import Any


class Sam2MarkerAdapter:
    def __init__(self, generator: Any | None = None):
        self._generator = generator

    def is_available(self) -> bool:
        generator = self._get_generator()
        return generator is not None and generator.is_available()

    def generate_markers(self, image):
        generator = self._get_generator()
        if generator is None or not generator.is_available():
            raise RuntimeError("SAM2 adapter is not available")
        return generator.generate_markers(image)

    def _get_generator(self):
        if self._generator is not None:
            return self._generator

        try:
            from sam2_marker_generator import build_sam2_marker_generator
        except ImportError:
            return None

        self._generator = build_sam2_marker_generator()
        return self._generator
