import cv2
import numpy as np


def _odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def denoise(image, method="median", strength=3):
    if method == "none":
        return image.copy()

    if method == "median":
        return cv2.medianBlur(image, _odd(strength))

    if method == "nlm":
        h = float(max(1, strength))
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

    raise ValueError(f"Bo loc khong ho tro: {method}")


def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if clip_limit <= 0:
        return image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = clahe.apply(lightness)
    merged = cv2.merge((lightness, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def adjust_gamma(image, gamma=1.0):
    if gamma <= 0:
        raise ValueError("Gamma phai lon hon 0")
    if abs(gamma - 1.0) < 1e-9:
        return image.copy()

    inverse = 1.0 / float(gamma)
    table = np.array(
        [((index / 255.0) ** inverse) * 255 for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def unsharp_mask(image, amount=0.0, radius=3.0):
    if amount <= 0:
        return image.copy()

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=float(radius))
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def upscale_lanczos(image, scale=2):
    scale = int(scale)
    if scale < 1:
        raise ValueError("He so phong to phai >= 1")
    if scale == 1:
        return image.copy()

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LANCZOS4,
    )


def preprocess(
    image,
    denoise_method="median",
    denoise_strength=3,
    clahe_clip=2.0,
    gamma=1.0,
):
    result = denoise(image, method=denoise_method, strength=denoise_strength)
    result = enhance_contrast(result, clip_limit=clahe_clip)
    result = adjust_gamma(result, gamma=gamma)
    return result
