import unittest

try:
    import numpy as np
except ImportError:
    np = None

try:
    from src.preview import comparison_html
except ImportError:
    comparison_html = None


@unittest.skipIf(np is None or comparison_html is None, "Thieu OpenCV")
class PreviewTests(unittest.TestCase):
    def test_comparison_html_has_slider_and_images(self):
        before = np.zeros((40, 60, 3), dtype=np.uint8)
        after = np.full((40, 60, 3), 200, dtype=np.uint8)
        html, height = comparison_html(before, after)
        self.assertIn('type="range"', html)
        self.assertIn("data:image/png;base64,", html)
        self.assertNotIn("__BEFORE__", html)
        self.assertNotIn("__AFTER__", html)
        self.assertGreater(height, 0)


if __name__ == "__main__":
    unittest.main()
