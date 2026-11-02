import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

if cv2 is not None:
    from src.inpainting import inpaint
    from src.preprocessing import (
        adjust_gamma,
        denoise,
        enhance_contrast,
        preprocess,
        unsharp_mask,
        upscale_lanczos,
    )


@unittest.skipIf(cv2 is None, "OpenCV chua duoc cai dat")
class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.image = rng.integers(
            0,
            256,
            size=(48, 64, 3),
            dtype=np.uint8,
        )

    def test_denoise_preserves_shape(self):
        for method in ("none", "median", "nlm"):
            result = denoise(self.image, method=method, strength=3)
            self.assertEqual(result.shape, self.image.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_contrast_and_gamma_keep_shape(self):
        contrast = enhance_contrast(self.image, clip_limit=2.0)
        adjusted = adjust_gamma(contrast, gamma=0.8)
        self.assertEqual(adjusted.shape, self.image.shape)
        self.assertEqual(adjusted.dtype, np.uint8)

    def test_preprocess_pipeline(self):
        result = preprocess(
            self.image,
            denoise_method="median",
            denoise_strength=3,
            clahe_clip=2.0,
            gamma=1.0,
        )
        self.assertEqual(result.shape, self.image.shape)

    def test_upscale_and_sharpen(self):
        upscaled = upscale_lanczos(self.image, scale=2)
        sharpened = unsharp_mask(upscaled, amount=0.5)
        self.assertEqual(upscaled.shape[:2], (96, 128))
        self.assertEqual(sharpened.shape, upscaled.shape)

    def test_inpaint_with_mask(self):
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask[10:20, 10:20] = 255
        result = inpaint(self.image, mask, radius=3)
        self.assertEqual(result.shape, self.image.shape)


if __name__ == "__main__":
    unittest.main()
