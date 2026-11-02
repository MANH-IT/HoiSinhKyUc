import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
COLORIZATION_DIR = MODEL_DIR / "colorization"

MODELS = {
    "gfpgan": (
        "GFPGANv1.4.pth",
        "https://github.com/TencentARC/GFPGAN/releases/download/"
        "v1.3.0/GFPGANv1.4.pth",
    ),
    "realesrgan": (
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth",
    ),
}

COLORIZATION_MODELS = [
    (
        "colorization_deploy_v2.prototxt",
        "https://raw.githubusercontent.com/richzhang/colorization/"
        "caffe/colorization/models/colorization_deploy_v2.prototxt",
    ),
    (
        "colorization_release_v2.caffemodel",
        "https://github.com/spmallick/learnopencv/releases/download/"
        "Colorization/colorization_release_v2.caffemodel",
    ),
    (
        "pts_in_hull.npy",
        "https://raw.githubusercontent.com/richzhang/colorization/"
        "caffe/colorization/resources/pts_in_hull.npy",
    ),
]


def download(url, destination, force=False):
    destination = Path(destination)
    if destination.exists() and not force:
        print(f"Bo qua (da co): {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "hoi-sinh-ky-uc/1.0"},
    )

    print(f"Dang tai: {url}")
    print(f" -> {destination}")
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(destination, "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = downloaded * 100 / total
                    print(
                        f"\r  {downloaded / 1048576:.1f} MB / "
                        f"{total / 1048576:.1f} MB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
    print()
    print(f"Xong: {destination}")


def build_parser():
    parser = argparse.ArgumentParser(description="Tai model AI cho du an")
    parser.add_argument("--all", action="store_true", help="Tai tat ca model")
    parser.add_argument("--gfpgan", action="store_true", help="Tai GFPGAN")
    parser.add_argument(
        "--realesrgan",
        action="store_true",
        help="Tai Real-ESRGAN",
    )
    parser.add_argument(
        "--colorization",
        action="store_true",
        help="Tai model to mau Caffe",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tai lai du file da ton tai",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected = []
    if args.all:
        selected = list(MODELS) + ["colorization"]
    else:
        if args.gfpgan:
            selected.append("gfpgan")
        if args.realesrgan:
            selected.append("realesrgan")
        if args.colorization:
            selected.append("colorization")
        if not selected:
            selected = list(MODELS) + ["colorization"]

    for name in selected:
        try:
            if name == "colorization":
                for filename, url in COLORIZATION_MODELS:
                    download(
                        url,
                        COLORIZATION_DIR / filename,
                        force=args.force,
                    )
            else:
                filename, url = MODELS[name]
                download(url, MODEL_DIR / filename, force=args.force)
        except Exception as exc:
            print(f"Loi khi tai {name}: {exc}", file=sys.stderr)
            return 1

    print("Hoan tat. Kiem tra thu muc models/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
