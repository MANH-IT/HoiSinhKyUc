from pathlib import Path

import cv2
import numpy as np


def load_image(path, grayscale=False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay anh: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"File anh rong hoac khong doc duoc: {path}")

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Khong giai ma duoc anh: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extension = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Khong ghi duoc dinh dang {extension}: {path}")

    encoded.tofile(str(path))
    return path


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def to_rgb(image):
    return cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2RGB)
