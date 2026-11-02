import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, restore_image


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phuc che di anh liet si / chien si",
    )
    parser.add_argument("--input", required=True, help="Duong dan anh goc")
    parser.add_argument("--output", required=True, help="Duong dan anh ket qua")
    parser.add_argument(
        "--denoise",
        choices=["none", "median", "nlm"],
        default="median",
        help="Bo loc khu nhieu",
    )
    parser.add_argument(
        "--denoise-strength",
        type=int,
        default=3,
        help="Kich thuoc median hoac cuong do NLM",
    )
    parser.add_argument(
        "--clahe",
        type=float,
        default=2.0,
        help="Cuong do CLAHE, 0 de tat",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--inpaint-mask", default=None)
    parser.add_argument("--inpaint-radius", type=int, default=3)
    parser.add_argument(
        "--inpaint-method",
        choices=["telea", "ns"],
        default="telea",
    )
    parser.add_argument("--upscale", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--sharpen", type=float, default=0.0)
    parser.add_argument("--face-restore", action="store_true")
    parser.add_argument("--fidelity", type=float, default=0.7)
    parser.add_argument("--colorize", action="store_true")
    parser.add_argument(
        "--colorize-engine",
        choices=["caffe", "deoldify"],
        default="caffe",
        help="Bo to mau: caffe (nhe) hoac deoldify (cao cap)",
    )
    parser.add_argument("--color-strength", type=float, default=0.85)
    parser.add_argument("--color-saturation", type=float, default=1.0)
    parser.add_argument("--use-ai-upscale", action="store_true")
    parser.add_argument("--gfpgan-model", default=None)
    parser.add_argument("--realesrgan-model", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = PipelineConfig(
        denoise=args.denoise,
        denoise_strength=args.denoise_strength,
        clahe=args.clahe,
        gamma=args.gamma,
        inpaint_mask=args.inpaint_mask,
        inpaint_radius=args.inpaint_radius,
        inpaint_method=args.inpaint_method,
        upscale=args.upscale,
        sharpen=args.sharpen,
        face_restore=args.face_restore,
        fidelity=args.fidelity,
        colorize=args.colorize,
        colorize_engine=args.colorize_engine,
        color_strength=args.color_strength,
        color_saturation=args.color_saturation,
        use_ai_upscale=args.use_ai_upscale,
    )
    if args.gfpgan_model:
        config.gfpgan_model = args.gfpgan_model
    if args.realesrgan_model:
        config.realesrgan_model = args.realesrgan_model

    try:
        result = restore_image(args.input, args.output, config)
    except Exception as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 1

    height, width = result.shape[:2]
    print(f"Da luu: {args.output}")
    print(f"Kich thuoc: {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
