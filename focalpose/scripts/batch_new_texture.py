import os
import shutil
from pathlib import Path

# === User-editable constants ===========
SRC_ROOT = Path("/mnt/c/Users/whu082/Downloads/Splited")
DST_ROOT = Path("/mnt/c/Users/whu082/focalpose/local_data/texture_datasets/texture_dataset")
# =======================================

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def copy_and_rename(src_root: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    counter = 1

    for root, _, files in os.walk(src_root):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext in IMAGE_EXTS:
                src = Path(root) / fname
                new_name = f"{counter:08d}{ext}"
                dst = dst_root / new_name
                shutil.copy2(src, dst)
                print(f"Copied: {src} → {dst}")
                counter += 1

    print(f"\nDone. {counter - 1} files copied to {dst_root}.")

if __name__ == "__main__":
    copy_and_rename(SRC_ROOT, DST_ROOT)
