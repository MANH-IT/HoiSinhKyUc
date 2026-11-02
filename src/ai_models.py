import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


class ModelUnavailable(RuntimeError):
    pass


class FaceRestorer:
    def __init__(self, model_path, upscale=1):
        self.model_path = Path(model_path)
        self.upscale = int(upscale)

    def restore(self, image_bgr, fidelity_weight=0.7):
        if not self.model_path.exists():
            raise ModelUnavailable(
                f"Khong tim thay model GFPGAN: {self.model_path}"
            )

        try:
            from gfpgan import GFPGANer
        except ImportError as exc:
            raise ModelUnavailable(
                "Chua cai GFPGAN. Chay: pip install -r requirements-ai.txt"
            ) from exc

        restorer = GFPGANer(
            model_path=str(self.model_path),
            upscale=self.upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _, _, restored = restorer.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=float(fidelity_weight),
        )
        if restored is None:
            raise RuntimeError("GFPGAN khong tim thay khuon mat de phuc hoi")
        return restored


class RealESRGANUpscaler:
    def __init__(self, model_path, scale=4, tile=0, half=False):
        self.model_path = Path(model_path)
        self.scale = int(scale)
        self.tile = int(tile)
        self.half = bool(half)

    def upscale(self, image_bgr, outscale=4):
        if not self.model_path.exists():
            raise ModelUnavailable(
                f"Khong tim thay model Real-ESRGAN: {self.model_path}"
            )

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise ModelUnavailable(
                "Chua cai Real-ESRGAN. Chay: pip install -r requirements-ai.txt"
            ) from exc

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale,
        )
        upsampler = RealESRGANer(
            scale=self.scale,
            model_path=str(self.model_path),
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=self.half,
        )
        output, _ = upsampler.enhance(image_bgr, outscale=int(outscale))
        return output


@lru_cache(maxsize=2)
def _load_caffe_colorizer(prototxt_path, caffemodel_path, pts_path):
    paths = [Path(prototxt_path), Path(caffemodel_path), Path(pts_path)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ModelUnavailable(
            "Chua tai model to mau. Thieu: " + ", ".join(missing)
        )

    net = cv2.dnn.readNetFromCaffe(str(paths[0]), str(paths[1]))
    pts = np.load(str(paths[2])).transpose().reshape(2, 313, 1, 1)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [
        np.full([1, 313], 2.606, dtype="float32")
    ]
    return net


class CaffeColorizer:
    def __init__(
        self,
        prototxt_path,
        caffemodel_path,
        pts_path,
        color_strength=0.85,
        saturation=1.0,
    ):
        self.prototxt_path = Path(prototxt_path)
        self.caffemodel_path = Path(caffemodel_path)
        self.pts_path = Path(pts_path)
        self.color_strength = float(np.clip(color_strength, 0.0, 1.0))
        self.saturation = max(0.0, float(saturation))

    def colorize(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Anh dau vao rong")

        image_bgr = np.asarray(image_bgr)
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        if image_bgr.dtype != np.uint8:
            image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

        net = _load_caffe_colorizer(
            str(self.prototxt_path),
            str(self.caffemodel_path),
            str(self.pts_path),
        )

        scaled = image_bgr.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_lab = cv2.cvtColor(
            gray_bgr.astype("float32") / 255.0,
            cv2.COLOR_BGR2LAB,
        )
        resized = cv2.resize(gray_lab, (224, 224))
        lightness = resized[:, :, 0] - 50

        net.setInput(cv2.dnn.blobFromImage(lightness))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image_bgr.shape[1], image_bgr.shape[0]))

        orig_l = lab[:, :, 0]
        orig_a = lab[:, :, 1]
        orig_b = lab[:, :, 2]

        new_a = orig_a + (ab[:, :, 0] - orig_a) * self.color_strength
        new_b = orig_b + (ab[:, :, 1] - orig_b) * self.color_strength
        new_a = np.clip(new_a * self.saturation, -127.0, 127.0)
        new_b = np.clip(new_b * self.saturation, -127.0, 127.0)

        merged = np.dstack([orig_l, new_a, new_b]).astype("float32")
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)


class DeOldifyColorizer:
    def __init__(self, artistic=True, render_factor=35):
        self.artistic = bool(artistic)
        self.render_factor = int(render_factor)

    def colorize(self, image_bgr):
        try:
            from deoldify.visualize import get_image_colorizer
            from PIL import Image
        except ImportError as exc:
            raise ModelUnavailable(
                "Chua cai DeOldify. Xem huong dan cai dat trong README."
            ) from exc

        colorizer = get_image_colorizer(artistic=self.artistic)
        source = Image.fromarray(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        )
        result = colorizer.get_transformed_image(
            source,
            render_factor=self.render_factor,
            watermarked=False,
        )
        if result is None:
            raise RuntimeError("DeOldify khong tao duoc anh mau")

        array = np.asarray(result.convert("RGB"))
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
