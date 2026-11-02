import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

if cv2 is not None:
    from src.ai_models import (
        CaffeColorizer,
        FaceRestorer,
        ModelUnavailable,
        RealESRGANUpscaler,
    )


@unittest.skipIf(cv2 is None, "OpenCV chua duoc cai dat")
class AIModelTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((32, 32, 3), 128, dtype=np.uint8)

    def test_missing_gfpgan_model_raises(self):
        restorer = FaceRestorer("/tmp/khong_ton_tai_gfpgan.pth")
        with self.assertRaises(ModelUnavailable):
            restorer.restore(self.image)

    def test_missing_realesrgan_model_raises(self):
        upscaler = RealESRGANUpscaler("/tmp/khong_ton_tai_realesrgan.pth")
        with self.assertRaises(ModelUnavailable):
            upscaler.upscale(self.image)

    def test_missing_colorization_model_raises(self):
        colorizer = CaffeColorizer(
            "/tmp/khong_ton_tai.prototxt",
            "/tmp/khong_ton_tai.caffemodel",
            "/tmp/khong_ton_tai.npy",
        )
        with self.assertRaises(ModelUnavailable):
            colorizer.colorize(self.image)


if __name__ == "__main__":
    unittest.main()
