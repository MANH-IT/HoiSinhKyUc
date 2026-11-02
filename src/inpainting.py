import cv2
import numpy as np


def dilate_mask(mask, iterations=1, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=max(0, int(iterations)))


def inpaint(image, mask, radius=3, method="telea"):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("Mask phai cung kich thuoc voi anh")

    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary_mask):
        return image.copy()

    flags = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS,
    }
    if method not in flags:
        raise ValueError(f"Phuong phap inpaint khong ho tro: {method}")

    return cv2.inpaint(
        image,
        binary_mask,
        float(radius),
        flags[method],
    )


def mask_from_dark_defects(image, threshold=35, dilate=1):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
    if dilate > 0:
        mask = dilate_mask(mask, iterations=dilate)
    return mask


def mask_from_light_defects(image, threshold=240, dilate=1):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
    if dilate > 0:
        mask = dilate_mask(mask, iterations=dilate)
    return mask
