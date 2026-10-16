import math
from dataclasses import dataclass

import cv2
import numpy as np

from backend.contracts.registration_models import (
    CropRect,
    RegistrationConfig,
    RegistrationDiagnostics,
    RegistrationResult,
)


@dataclass(frozen=True)
class RegistrationProxy:
    before: np.ndarray
    after: np.ndarray
    before_mask: np.ndarray
    after_mask: np.ndarray
    scale: float


@dataclass(frozen=True)
class FeatureMatchResult:
    points_after: np.ndarray
    points_before: np.ndarray
    inlier_mask: np.ndarray
    keypoints_before: int
    keypoints_after: int
    good_matches: int
    scale_hint: float | None


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Изображение не задано или пустое")
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Неподдерживаемая форма изображения: {image.shape}")
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Неподдерживаемое число каналов: {channels}")


def _normalize_u8(image: np.ndarray, use_clahe: bool) -> np.ndarray:
    gray = _to_grayscale(image).astype(np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError("Изображение не содержит конечных значений")
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        raise ValueError("Изображение практически константное; регистрация невозможна")
    normalized = np.clip((gray - low) / (high - low), 0.0, 1.0)
    u8 = np.round(normalized * 255.0).astype(np.uint8)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        u8 = clahe.apply(u8)
    return u8


def _validate_bottom_exclusion(height: int, bottom_exclusion_px: int) -> int:
    value = int(round(bottom_exclusion_px or 0))
    if value < 0:
        raise ValueError("Отступ снизу не может быть отрицательным")
    # At least 32 px of the scientific image must remain. This avoids obscure
    # SIFT/ECC failures when a value larger than image height is entered.
    max_allowed = max(0, height - 32)
    if value > max_allowed:
        raise ValueError(
            "Слишком большой отступ снизу: после исключения легенды должно остаться "
            "не менее 32 пикселей изображения"
        )
    return value


def _make_content_mask(shape: tuple[int, ...], bottom_exclusion_px: int) -> np.ndarray:
    height, width = shape[:2]
    exclusion = _validate_bottom_exclusion(height, bottom_exclusion_px)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    if exclusion > 0:
        mask[height - exclusion :, :] = 0
    return mask


def build_registration_proxy(
    before: np.ndarray,
    after: np.ndarray,
    config: RegistrationConfig,
) -> RegistrationProxy:
    before_u8 = _normalize_u8(before, config.use_clahe)
    after_u8 = _normalize_u8(after, config.use_clahe)
    before_mask = _make_content_mask(before.shape, config.bottom_exclusion_px)
    after_mask = _make_content_mask(after.shape, config.bottom_exclusion_px)

    max_dimension = max(
        before_u8.shape[0],
        before_u8.shape[1],
        after_u8.shape[0],
        after_u8.shape[1],
    )
    scale = min(1.0, config.proxy_max_side / float(max_dimension))

    def resize_image(image: np.ndarray) -> np.ndarray:
        if scale == 1.0:
            return image
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def resize_mask(mask: np.ndarray) -> np.ndarray:
        if scale == 1.0:
            return mask
        width = max(1, int(round(mask.shape[1] * scale)))
        height = max(1, int(round(mask.shape[0] * scale)))
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    before_proxy = resize_image(before_u8)
    after_proxy = resize_image(after_u8)
    before_mask_proxy = resize_mask(before_mask)
    after_mask_proxy = resize_mask(after_mask)

    canvas_h = max(before_proxy.shape[0], after_proxy.shape[0])
    canvas_w = max(before_proxy.shape[1], after_proxy.shape[1])

    def pad_image(image: np.ndarray) -> np.ndarray:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        return canvas

    def pad_mask(mask: np.ndarray) -> np.ndarray:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas[: mask.shape[0], : mask.shape[1]] = mask
        return canvas

    return RegistrationProxy(
        before=pad_image(before_proxy),
        after=pad_image(after_proxy),
        before_mask=pad_mask(before_mask_proxy),
        after_mask=pad_mask(after_mask_proxy),
        scale=scale,
    )


def match_features(
    fixed: np.ndarray,
    moving: np.ndarray,
    config: RegistrationConfig,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
) -> FeatureMatchResult:
    sift = cv2.SIFT_create(nfeatures=config.sift_nfeatures)
    keypoints_before, descriptors_before = sift.detectAndCompute(fixed, fixed_mask)
    keypoints_after, descriptors_after = sift.detectAndCompute(moving, moving_mask)

    count_before = len(keypoints_before or [])
    count_after = len(keypoints_after or [])
    if descriptors_before is None or descriptors_after is None:
        raise ValueError(
            "Недостаточно локальных признаков в рабочей области для автоматического совмещения"
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(descriptors_after, descriptors_before, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < config.lowe_ratio * second.distance:
            good.append(first)

    if len(good) < config.min_good_matches:
        raise ValueError(
            "Недостаточно устойчивых соответствий в рабочей области: "
            f"{len(good)} < {config.min_good_matches}. Попробуйте уменьшить отступ снизу."
        )

    points_after = np.float32([keypoints_after[m.queryIdx].pt for m in good])
    points_before = np.float32([keypoints_before[m.trainIdx].pt for m in good])

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_after,
        points_before,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        maxIters=5000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or inlier_mask is None:
        raise ValueError("RANSAC не смог оценить геометрическое соответствие изображений")

    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 3:
        raise ValueError("Недостаточно согласованных соответствий после RANSAC")

    a, b = float(affine[0, 0]), float(affine[0, 1])
    scale_hint = math.sqrt(a * a + b * b)

    return FeatureMatchResult(
        points_after=points_after,
        points_before=points_before,
        inlier_mask=inlier_mask,
        keypoints_before=count_before,
        keypoints_after=count_after,
        good_matches=len(good),
        scale_hint=scale_hint,
    )


def estimate_rigid_transform(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
) -> np.ndarray:
    moving = np.asarray(moving_points, dtype=np.float64)
    fixed = np.asarray(fixed_points, dtype=np.float64)
    if moving.shape != fixed.shape or moving.ndim != 2 or moving.shape[1] != 2:
        raise ValueError("Точки должны иметь одинаковую форму Nx2")
    if moving.shape[0] < 2:
        raise ValueError("Для оценки rigid transform нужны минимум две пары точек")

    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    x = moving - moving_center
    y = fixed - fixed_center

    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = fixed_center - rotation @ moving_center
    transform = np.zeros((2, 3), dtype=np.float64)
    transform[:, :2] = rotation
    transform[:, 2] = translation
    return transform.astype(np.float32)


def _common_proxy_mask(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    direct_after_to_before: np.ndarray,
) -> np.ndarray:
    height, width = fixed_mask.shape[:2]
    warped_moving_mask = cv2.warpAffine(
        moving_mask,
        direct_after_to_before,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    common = cv2.bitwise_and(fixed_mask, warped_moving_mask)
    # Remove the artificial hard edge around the excluded footer/padding from
    # the ECC objective. It should align sample texture, not the ROI boundary.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    common = cv2.erode(common, kernel, iterations=1)
    if cv2.countNonZero(common) == 0:
        raise ValueError("После исключения нижней области не осталось общей зоны для ECC")
    return common


def refine_rigid_ecc(
    fixed: np.ndarray,
    moving: np.ndarray,
    initial_after_to_before: np.ndarray,
    config: RegistrationConfig,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    fixed_float = fixed.astype(np.float32) / 255.0
    moving_float = moving.astype(np.float32) / 255.0

    ecc_mask = None
    if fixed_mask is not None and moving_mask is not None:
        ecc_mask = _common_proxy_mask(
            fixed_mask,
            moving_mask,
            initial_after_to_before,
        )

    # findTransformECC returns a matrix used with WARP_INVERSE_MAP in the
    # canonical OpenCV example. Our public contract is the opposite:
    # direct transform AFTER -> BEFORE. Therefore ECC is seeded in inverse form
    # and inverted again before returning.
    ecc_warp = cv2.invertAffineTransform(initial_after_to_before).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_max_iterations,
        config.ecc_epsilon,
    )
    score, ecc_warp = cv2.findTransformECC(
        templateImage=fixed_float,
        inputImage=moving_float,
        warpMatrix=ecc_warp,
        motionType=cv2.MOTION_EUCLIDEAN,
        criteria=criteria,
        inputMask=ecc_mask,
        gaussFiltSize=config.ecc_gauss_filter_size,
    )
    direct = cv2.invertAffineTransform(ecc_warp).astype(np.float32)
    return direct, float(score)


def _transform_proxy_to_original(transform: np.ndarray, scale: float) -> np.ndarray:
    original = np.asarray(transform, dtype=np.float32).copy()
    original[:, 2] /= float(scale)
    return original


def _rotation_deg(transform: np.ndarray) -> float:
    return math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))


def _warp_image(
    image: np.ndarray,
    transform: np.ndarray,
    output_width: int,
    output_height: int,
    interpolation: int,
) -> np.ndarray:
    return cv2.warpAffine(
        image,
        transform,
        (output_width, output_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compute_overlap_mask(
    before_shape: tuple[int, ...],
    after_shape: tuple[int, ...],
    transform_after_to_before: np.ndarray,
    bottom_exclusion_px: int = 0,
) -> np.ndarray:
    before_h, before_w = before_shape[:2]
    before_valid = _make_content_mask(before_shape, bottom_exclusion_px)
    after_valid = _make_content_mask(after_shape, bottom_exclusion_px)
    warped_after_valid = _warp_image(
        after_valid,
        transform_after_to_before,
        output_width=before_w,
        output_height=before_h,
        interpolation=cv2.INTER_NEAREST,
    )
    return ((before_valid > 0) & (warped_after_valid > 0)).astype(np.uint8)


def largest_valid_rectangle(mask: np.ndarray) -> CropRect:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError("Overlap mask должна быть непустой двумерной матрицей")

    heights = np.zeros(binary.shape[1], dtype=np.int32)
    best_area = 0
    best_rect = None

    for row in range(binary.shape[0]):
        heights = np.where(binary[row] > 0, heights + 1, 0)
        stack: list[int] = []
        for col in range(binary.shape[1] + 1):
            current = int(heights[col]) if col < binary.shape[1] else 0
            while stack and current < int(heights[stack[-1]]):
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                width = col - left
                area = height * width
                if area > best_area:
                    best_area = area
                    best_rect = CropRect(
                        x=left,
                        y=row - height + 1,
                        width=width,
                        height=height,
                    )
            stack.append(col)

    if best_rect is None or best_area == 0:
        raise ValueError("Изображения не имеют общей валидной области")
    return best_rect


def _make_overlay(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_gray = _normalize_u8(before, use_clahe=False)
    after_gray = _normalize_u8(after, use_clahe=False)
    overlay = np.zeros((*before_gray.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = before_gray
    overlay[:, :, 1] = after_gray
    overlay[:, :, 2] = after_gray
    return overlay


def register_image_pair(
    before: np.ndarray,
    after: np.ndarray,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    config = config or RegistrationConfig()
    if before is None or after is None:
        raise ValueError("Необходимо загрузить оба изображения: до и после")
    if before.size == 0 or after.size == 0:
        raise ValueError("Загружено пустое изображение")

    # Validate against each image separately because the same footer height is
    # excluded from both source rasters.
    _validate_bottom_exclusion(before.shape[0], config.bottom_exclusion_px)
    _validate_bottom_exclusion(after.shape[0], config.bottom_exclusion_px)

    proxy = build_registration_proxy(before, after, config)
    matches = match_features(
        proxy.before,
        proxy.after,
        config,
        fixed_mask=proxy.before_mask,
        moving_mask=proxy.after_mask,
    )

    inliers_after = matches.points_after[matches.inlier_mask]
    inliers_before = matches.points_before[matches.inlier_mask]
    rigid_proxy = estimate_rigid_transform(inliers_after, inliers_before)

    warnings: list[str] = []
    if config.bottom_exclusion_px > 0:
        warnings.append(
            f"Нижние {config.bottom_exclusion_px} px обоих изображений исключены "
            "из регистрации и итоговой общей области."
        )
    if (
        matches.scale_hint is not None
        and abs(matches.scale_hint - 1.0) > config.scale_warning_threshold
    ):
        warnings.append(
            "Оценка соответствий указывает на возможное различие масштаба; "
            "масштабирование намеренно не применялось."
        )

    ecc_score = None
    final_proxy = rigid_proxy
    try:
        final_proxy, ecc_score = refine_rigid_ecc(
            proxy.before,
            proxy.after,
            rigid_proxy,
            config,
            fixed_mask=proxy.before_mask,
            moving_mask=proxy.after_mask,
        )
    except (cv2.error, ValueError):
        warnings.append(
            "ECC-уточнение не сошлось; использовано rigid-совмещение по локальным признакам."
        )

    transform = _transform_proxy_to_original(final_proxy, proxy.scale)
    before_h, before_w = before.shape[:2]
    after_warped = _warp_image(
        after,
        transform,
        output_width=before_w,
        output_height=before_h,
        interpolation=cv2.INTER_LINEAR,
    )

    # The footer/scale legend is excluded not only from matching but from the
    # downloadable common crop as well. For the moving image the exclusion mask
    # is warped with the same transform, so the crop cannot accidentally include
    # the transformed legend after rotation/translation.
    overlap_mask = compute_overlap_mask(
        before.shape,
        after.shape,
        transform,
        bottom_exclusion_px=config.bottom_exclusion_px,
    )
    crop = largest_valid_rectangle(overlap_mask)
    y0, y1 = crop.y, crop.y + crop.height
    x0, x1 = crop.x, crop.x + crop.width
    before_aligned = np.ascontiguousarray(before[y0:y1, x0:x1])
    after_aligned = np.ascontiguousarray(after_warped[y0:y1, x0:x1])

    before_content_area = before.shape[1] * (
        before.shape[0] - config.bottom_exclusion_px
    )
    after_content_area = after.shape[1] * (
        after.shape[0] - config.bottom_exclusion_px
    )
    min_area = min(before_content_area, after_content_area)
    overlap_ratio = float(crop.width * crop.height) / float(min_area)
    inlier_count = int(matches.inlier_mask.sum())
    inlier_ratio = inlier_count / float(matches.good_matches)

    diagnostics = RegistrationDiagnostics(
        rotation_deg=_rotation_deg(transform),
        translation_x_px=float(transform[0, 2]),
        translation_y_px=float(transform[1, 2]),
        scale_hint=matches.scale_hint,
        keypoints_before=matches.keypoints_before,
        keypoints_after=matches.keypoints_after,
        good_matches=matches.good_matches,
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        ecc_score=ecc_score,
        overlap_ratio=overlap_ratio,
        bottom_exclusion_px=config.bottom_exclusion_px,
        warnings=warnings,
    )

    return RegistrationResult(
        before_aligned=before_aligned,
        after_aligned=after_aligned,
        overlay_preview=_make_overlay(before_aligned, after_aligned),
        transform_matrix=transform,
        crop_rect=crop,
        diagnostics=diagnostics,
    )
