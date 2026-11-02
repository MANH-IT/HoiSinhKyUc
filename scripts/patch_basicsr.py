import importlib.util
from pathlib import Path

OLD_IMPORT = (
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
)
NEW_IMPORT = "from torchvision.transforms.functional import rgb_to_grayscale"
OLD_MODULE = "torchvision.transforms.functional_tensor"
NEW_MODULE = "torchvision.transforms.functional"


def main():
    spec = importlib.util.find_spec("basicsr")
    if spec is None or not spec.submodule_search_locations:
        print("Khong tim thay basicsr trong moi truong hien tai.")
        return 1

    package_root = Path(list(spec.submodule_search_locations)[0])
    changed = 0

    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if OLD_MODULE not in text:
            continue

        updated = text.replace(OLD_IMPORT, NEW_IMPORT)
        updated = updated.replace(OLD_MODULE, NEW_MODULE)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
            print(f"Da va: {path}")

    print(f"Hoan tat. Da sua {changed} file trong {package_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
