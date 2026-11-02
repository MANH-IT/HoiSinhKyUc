from dataclasses import dataclass
from pathlib import Path

from config.settings import (
    COLORIZATION_CAFFEMODEL,
    COLORIZATION_PROTOTXT,
    COLORIZATION_PTS,
    DEFAULT_COLOR_SATURATION,
    DEFAULT_COLOR_STRENGTH,
    GFPGAN_MODEL,
    REALESRGAN_MODEL,
)
from src.ai_models import (
    CaffeColorizer,
    DeOldifyColorizer,
    FaceRestorer,
    RealESRGANUpscaler,
)
from src.inpainting import inpaint
from src.io_utils import load_image, save_image, to_bgr
from src.preprocessing import preprocess, unsharp_mask, upscale_lanczos


@dataclass
class PipelineConfig:
    denoise: str = "median"
    denoise_strength: int = 3
    clahe: float = 2.0
    gamma: float = 1.0
    inpaint_mask: str | None = None
    inpaint_radius: int = 3
    inpaint_method: str = "telea"
    upscale: int = 1
    sharpen: float = 0.0
    face_restore: bool = False
    fidelity: float = 0.7
    colorize: bool = False
    colorize_engine: str = "caffe"
    color_strength: float = DEFAULT_COLOR_STRENGTH
    color_saturation: float = DEFAULT_COLOR_SATURATION
    use_ai_upscale: bool = False
    gfpgan_model: str = str(GFPGAN_MODEL)
    realesrgan_model: str = str(REALESRGAN_MODEL)
    colorization_prototxt: str = str(COLORIZATION_PROTOTXT)
    colorization_caffemodel: str = str(COLORIZATION_CAFFEMODEL)
    colorization_pts: str = str(COLORIZATION_PTS)


def restore_array(image, config, inpaint_mask=None):
    result = preprocess(
        image,
        denoise_method=config.denoise,
        denoise_strength=config.denoise_strength,
        clahe_clip=config.clahe,
        gamma=config.gamma,
    )

    if inpaint_mask is not None:
        result = inpaint(
            result,
            inpaint_mask,
            radius=config.inpaint_radius,
            method=config.inpaint_method,
        )
    elif config.inpaint_mask:
        mask = load_image(config.inpaint_mask, grayscale=True)
        result = inpaint(
            result,
            mask,
            radius=config.inpaint_radius,
            method=config.inpaint_method,
        )

    if config.face_restore:
        result = FaceRestorer(config.gfpgan_model).restore(
            to_bgr(result),
            fidelity_weight=config.fidelity,
        )

    if config.colorize:
        engine = (config.colorize_engine or "caffe").lower()
        if engine == "deoldify":
            result = DeOldifyColorizer().colorize(to_bgr(result))
        elif engine == "caffe":
            result = CaffeColorizer(
                config.colorization_prototxt,
                config.colorization_caffemodel,
                config.colorization_pts,
                color_strength=config.color_strength,
                saturation=config.color_saturation,
            ).colorize(to_bgr(result))
        else:
            raise ValueError(
                "Bo to mau khong hop le: "
                f"{config.colorize_engine}. Chon 'caffe' hoac 'deoldify'."
            )

    if config.upscale > 1:
        if config.use_ai_upscale:
            result = RealESRGANUpscaler(config.realesrgan_model).upscale(
                to_bgr(result),
                outscale=config.upscale,
            )
        else:
            result = upscale_lanczos(result, scale=config.upscale)

    return unsharp_mask(result, amount=config.sharpen)


def restore_image(input_path, output_path, config):
    image = load_image(input_path)
    result = restore_array(image, config)
    save_image(output_path, result)
    return result
