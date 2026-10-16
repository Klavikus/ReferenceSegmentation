import cv2
import numpy as np


def validate_mask_payload(mask_payload: dict) -> None:
    if mask_payload is None:
        raise ValueError("Mask payload is missing")
    if "layers" not in mask_payload:
        raise ValueError("Mask payload must contain layers")
    if not mask_payload["layers"]:
        raise ValueError("Mask payload layers must not be empty")


def extract_first_mask_layer(mask_payload: dict) -> np.ndarray:
    validate_mask_payload(mask_payload)

    layer = np.array(mask_payload["layers"][0])
    if layer.ndim == 3 and layer.shape[2] == 4:
        return cv2.cvtColor(layer, cv2.COLOR_RGBA2GRAY)
    if layer.ndim == 3 and layer.shape[2] == 3:
        return cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    return layer.astype(np.uint8, copy=False)
