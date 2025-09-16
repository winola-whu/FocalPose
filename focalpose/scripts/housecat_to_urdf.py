#!/usr/bin/env python3
"""

For each object in every HouseCat category dir, create a destination folder
containing: <stem>.png, model.mtl, <stem>.mtl, <stem>.obj, model.obj, <stem>.urdf

- Copies originals (<stem>.*) as-is.
- Writes normalized model.obj / model.mtl that use relative paths
  and reference textures by filename only.
"""

import re
import shutil
from pathlib import Path

# ==== EDIT THESE CONSTANTS ====================================================
SRC_ROOT  = Path("/mnt/c/Users/whu082/HouseCat6D/obj_models_small_size_final")  # category subfolders (bottle/, cup/, ...)
DEST_ROOT = Path("/mnt/c/Users/whu082/focalpose/local_data/models_urdf/HouseCat")    # per-object folders
# ==============================================================================

MTL_MAP_KEYS = ("map_Kd", "map_Ka", "map_Ks", "map_d", "map_Bump", "bump")

def safe(s: str) -> str:
    """Lowercase and keep [a-z0-9.-], replace others with '_'."""
    return re.sub(r"[^a-z0-9._-]+", "_", s.lower())

def normalize_obj(obj_path: Path, has_mtl: bool) -> None:
    """
    Ensure 'model.obj' references 'model.mtl' (single mtllib line at top).
    Does NOT touch the original <stem>.obj.
    """
    text = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out, inserted, seen = [], False, False
    for ln in text:
        if ln.strip().startswith("mtllib"):
            if not seen:
                seen = True
                if has_mtl:
                    out.append("mtllib model.mtl")
            # drop all other mtllib lines
        else:
            out.append(ln)
    if has_mtl and not seen:
        out.insert(0, "mtllib model.mtl")
    obj_path.write_text("\n".join(out) + "\n", encoding="utf-8")

def parse_and_copy_textures(mtl_path: Path, src_dir: Path, dst_mesh_dir: Path) -> None:
    """
    Copy textures referenced by model.mtl into dst_mesh_dir and rewrite to filenames.
    """
    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        key = next((k for k in MTL_MAP_KEYS if stripped.startswith(k + " ")), None)
        if not key:
            out.append(line)
            continue

        toks = stripped.split()
        tex = next((t for t in reversed(toks[1:]) if not t.startswith("-")), None)
        if not tex:
            out.append(line)
            continue

        # try original relative path, then basename
        src_tex = (src_dir / tex)
        if not src_tex.exists():
            src_tex = src_dir / Path(tex).name

        if src_tex.exists():
            dst = dst_mesh_dir / src_tex.name
            if dst.resolve() != src_tex.resolve():
                shutil.copy2(src_tex, dst)
            out.append(f"{key} {dst.name}")
        else:
            out.append(line)

    mtl_path.write_text("\n".join(out) + "\n", encoding="utf-8")

def pack_one_category(cat_dir: Path) -> None:
    category = cat_dir.name
    urdfs = sorted(cat_dir.glob("*.urdf"))
    if not urdfs:
        print(f"[SKIP] {cat_dir.name}: no .urdf files")
        return

    for urdf in urdfs:
        stem = urdf.stem  # e.g., bottle-85_alcool
        obj_src = cat_dir / f"{stem}.obj"
        mtl_src = cat_dir / f"{stem}.mtl"
        png_src = cat_dir / f"{stem}_0.png"  # texture exported by Open3D etc.

        if not obj_src.exists():
            print(f"  [MISS OBJ] {stem}")
            continue

        obj_id   = safe(f"{stem}")
        obj_dest = DEST_ROOT / obj_id
        obj_dest.mkdir(parents=True, exist_ok=True)
        # mesh_dir = obj_dest / f"{stem}"
        # mesh_dir.mkdir(parents=True, exist_ok=True)

        # --- Copy originals into the object folder --------------------------------
        shutil.copy2(obj_src, obj_dest / f"{obj_id}.obj")
        if mtl_src.exists():
            shutil.copy2(mtl_src, obj_dest / f"{obj_id}.mtl")
        if png_src.exists():
            shutil.copy2(png_src, obj_dest / f"{obj_id}.png")
        shutil.copy2(urdf, obj_dest / f"{obj_id}.urdf")
        shutil.copy2(urdf, obj_dest / "model.urdf")

        # --- Create normalized model.obj / model.mtl + copy textures ---------------
        shutil.copy2(obj_src, obj_dest / "model.obj")
        has_mtl = mtl_src.exists()
        if has_mtl:
            shutil.copy2(mtl_src, obj_dest / "model.mtl")
            parse_and_copy_textures(obj_dest / "model.mtl", cat_dir, obj_dest)
        normalize_obj(obj_dest / "model.obj", has_mtl)

        print(f"  [OK] {obj_id}")

def main():
    print(f"SRC_ROOT : {SRC_ROOT}")
    print(f"DEST_ROOT: {DEST_ROOT}\n")
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for cat_dir in sorted(SRC_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        print(f"Processing category: {cat_dir.name}")
        pack_one_category(cat_dir)

    print("\nDone. Object folders written under:", DEST_ROOT)

if __name__ == "__main__":
    main()
