import cv2
import numpy as np

from backend.contracts.pipeline_models import ImageOffset, ImageSize, PreparedImageResult


def resize_and_pad_image(
    image: np.ndarray, target_width: int, target_height: int
) -> PreparedImageResult:
    if image is None:
        raise ValueError("Image is None")

    height, width = image.shape[:2]
    scale_factor = min(target_width / width, target_height / height)
    prepared_width = int(width * scale_factor)
    prepared_height = int(height * scale_factor)
    resized_image = cv2.resize(image, (prepared_width, prepared_height))

    x_offset = (target_width - prepared_width) // 2
    y_offset = (target_height - prepared_height) // 2

    if len(image.shape) == 2:
        prepared_image = np.full((target_height, target_width), 255, dtype=image.dtype)
        prepared_image[
            y_offset : y_offset + prepared_height,
            x_offset : x_offset + prepared_width,
        ] = resized_image
        grayscale_preview = cv2.cvtColor(prepared_image, cv2.COLOR_GRAY2RGB)
    else:
        prepared_image = np.full(
            (target_height, target_width, image.shape[2]), 255, dtype=image.dtype
        )
        prepared_image[
            y_offset : y_offset + prepared_height,
            x_offset : x_offset + prepared_width,
            :,
        ] = resized_image
        grayscale = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2GRAY)
        grayscale_preview = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

    return PreparedImageResult(
        prepared_image=prepared_image,
        grayscale_preview=grayscale_preview,
        prepared_size=ImageSize(width=prepared_width, height=prepared_height),
        prepared_offset=ImageOffset(x=x_offset, y=y_offset),
        scale_factor=scale_factor,
    )


def prepare_image(
    image: np.ndarray, target_width: int, target_height: int
) -> PreparedImageResult:
    return resize_and_pad_image(
        image=image,
        target_width=target_width,
        target_height=target_height,
    )
