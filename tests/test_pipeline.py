import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

if cv2 is not None:
    from src.pipeline import PipelineConfig, restore_array


@unittest.skipIf(cv2 is None, "OpenCV chua duoc cai dat")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.image = rng.integers(
            0,
            256,
            size=(40, 50, 3),
            dtype=np.uint8,
        )

    def test_restore_array_cpu(self):
        result = restore_array(
            self.image,
            PipelineConfig(clahe=0.0, upscale=2, sharpen=0.2),
        )
        self.assertEqual(result.shape[:2], (80, 100))

    def test_restore_array_with_mask_array(self):
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        mask[5:15, 5:15] = 255
        result = restore_array(
            self.image,
            PipelineConfig(clahe=0.0),
            inpaint_mask=mask,
        )
        self.assertEqual(result.shape, self.image.shape)

    def test_invalid_colorize_engine_raises(self):
        with self.assertRaises(ValueError):
            restore_array(
                self.image,
                PipelineConfig(
                    clahe=0.0,
                    colorize=True,
                    colorize_engine="khong_hop_le",
                ),
            )


if __name__ == "__main__":
    unittest.main()
