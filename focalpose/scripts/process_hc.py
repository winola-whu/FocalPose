#!/usr/bin/env python3
import argparse, json, os, sys, shutil, pickle, re
from pathlib import Path
from collections import defaultdict

# ---- constants (kept) ----
HOUSECAT_ROOT  = Path("/mnt/c/Users/whu082/focalpose/local_data/hc_test")
MODELS = Path("/mnt/c/Users/whu082/focalpose/local_data/HouseCat/models")
OUT_ROOT = Path("/mnt/c/Users/whu082/focalpose/local_data/HouseCat")

# ------------ helpers ------------
def read_intrinsics_txt(p: Path):
    """
    Accepts either 3x3 matrix (9 numbers) or 4 scalars (fx fy cx cy).
    Returns fx, fy, cx, cy as floats.
    """
    txt = p.read_text().strip().replace(",", " ")
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)]
    if len(nums) >= 9:  # 3x3
        fx, fy, cx, cy = nums[0], nums[4], nums[2], nums[5]
    elif len(nums) >= 4:
        fx, fy, cx, cy = nums[0], nums[1], nums[2], nums[3]
    else:
        raise ValueError(f"Unrecognized intrinsics format in {p}")
    return fx, fy, cx, cy

def parse_meta(scene_dir: Path):
    """
    Return a list of (inst_id_str, class_id_str, full_name_str) from meta.txt.
    Keeps the entire 3rd field as the instance name.
    """
    rows = []
    meta_file = scene_dir / "meta.txt"
    if not meta_file.exists():
        return rows
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            a, b, c = s.split(maxsplit=2)   # keep full 3rd field
            rows.append((a, b, c))
    return rows

def name_to_mesh_hint(name: str, meshes_root: Path):
    """
    Expect an instance folder '<name>/' containing '<name>.obj'.
    Return absolute POSIX path or None if missing.
    """
    # normalize separators/spaces
    safe = name.strip().replace("\\", "/")
    candidate = meshes_root / safe / f"{safe}.obj"
    return candidate.as_posix() if candidate.exists() else None

def safe_link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():  # idempotent
        return
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except Exception:
            pass
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def collect_frames(scene_dir: Path):
    """Robust discovery for RGB + labels (png/jpg + *_label.pkl / .pkl)."""
    rgb_candidates = [scene_dir / "RGB", scene_dir / "rgb", scene_dir / "images", scene_dir / "color"]
    rgb_dir = next((p for p in rgb_candidates if p.exists()), None)
    if rgb_dir is None:
        print(f"[WARN] No RGB dir in {scene_dir}", file=sys.stderr)
        return []

    label_candidates = [scene_dir / "labels", scene_dir / "label", scene_dir / "Labels",
                        scene_dir / "annotations", scene_dir / "gt"]
    labels_dir = next((p for p in label_candidates if p.exists()), None)
    if labels_dir is None:
        print(f"[WARN] No labels dir in {scene_dir}", file=sys.stderr)
        return []

    imgs = []
    for pat in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        imgs.extend(sorted(rgb_dir.glob(pat)))

    frames = []
    for img in imgs:
        fid = img.stem
        cand = [labels_dir / f"{fid}_label.pkl", labels_dir / f"{fid}.pkl"]
        pkl = next((c for c in cand if c.exists()), None)
        if pkl is None:
            continue
        frames.append((fid, img, pkl))

    print(f"[INFO] {scene_dir.name}: {len(frames)} frames (from {rgb_dir.relative_to(scene_dir)} / {labels_dir.relative_to(scene_dir)})")
    return frames

def load_label_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    return d

def dump(fp, obj):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser(description="Convert HouseCat scenes to a FocalPose-friendly real dataset.")
    ap.add_argument("--link_mode", default="hardlink", choices=["hardlink", "symlink", "copy"],
                    help="How to materialize images in the output tree")
    ap.add_argument("--tag", default="test", choices=["train", "val", "test"],
                    help="Write everything under this split (run once per split)")
    args = ap.parse_args()
    tag = args.tag

    out_images = OUT_ROOT / "images"
    out_ann    = OUT_ROOT / "annotations"
    out_meta   = OUT_ROOT / "meta"
    out_images.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    datasets = {tag: []}
    class_id_to_names_all = defaultdict(set)

    # Walk immediate subfolders; if none, treat root as one scene
    scene_dirs = [p for p in HOUSECAT_ROOT.iterdir() if p.is_dir()]
    if not scene_dirs:
        scene_dirs = [HOUSECAT_ROOT]

    for scene_dir in sorted(scene_dirs):
        scene_id = scene_dir.name
        split = tag

        intr_path = scene_dir / "intrinsics.txt"
        if not intr_path.exists():
            print(f"[WARN] Missing intrinsics in {scene_id}, skipping.", file=sys.stderr)
            continue
        fx, fy, cx, cy = read_intrinsics_txt(intr_path)

        # meta → dicts
        rows = parse_meta(scene_dir)
        inst2name, inst2cid = {}, {}
        cid2names = defaultdict(set)
        for inst_s, cid_s, name in rows:
            try: inst = int(inst_s)
            except: inst = inst_s
            try: cid = int(cid_s)
            except: cid = cid_s
            inst2name[inst] = name
            if isinstance(cid, int):
                cid2names[cid].add(name)
                class_id_to_names_all[cid].add(name)

        frames = collect_frames(scene_dir)
        if not frames:
            print(f"[WARN] No frames found in {scene_id}, skipping.", file=sys.stderr)
            continue

        for fid, img_path, lbl_path in frames:
            data = load_label_pkl(lbl_path)
            req = ["instance_ids", "class_ids", "bboxes", "translations", "rotations"]
            if not all(k in data for k in req):
                print(f"[WARN] Missing keys in {lbl_path}, has {list(data.keys())}", file=sys.stderr)
                continue

            dst_img = out_images / split / f"{scene_id}_{fid}{img_path.suffix.lower()}"
            safe_link_or_copy(img_path, dst_img, args.link_mode)

            rec = {
                "image_path": str(dst_img.relative_to(OUT_ROOT).as_posix()),
                "scene_id": scene_id,
                "frame_id": fid,
                "K": [float(fx), float(fy), float(cx), float(cy)],
                "objects": []
            }

            inst_ids = data["instance_ids"]; class_ids = data["class_ids"]
            bboxes = data["bboxes"]; trans = data["translations"]; rots = data["rotations"]
            N = min(len(inst_ids), len(class_ids), len(bboxes), len(trans), len(rots))
            for i in range(N):
                iid = int(inst_ids[i]); cid = int(class_ids[i])
                name = inst2name.get(iid)
                if name is None:
                    # fallback: any name for the class, else synthetic token
                    name = (sorted(list(cid2names.get(cid, set())))[:1] or [f"class{cid}_inst{iid}"])[0]

                bbox = [float(x) for x in bboxes[i]]
                R = rots[i]; R_list = [[float(R[r][c]) for c in range(3)] for r in range(3)]
                t = [float(x) for x in trans[i]]
                mesh_hint = name_to_mesh_hint(name, MODELS)

                rec["objects"].append({
                    "instance_id": iid,
                    "class_id": cid,
                    "name": name,          # full instance label from meta.txt
                    "bbox_xywh": bbox,
                    "R": R_list,
                    "t": t,
                    "model": mesh_hint     # absolute path or None
                })

            datasets[split].append(rec)

    dump(out_ann / f"housecat_{tag}.json", datasets[tag])
    dump(out_meta / "class_id_to_example_names.json",
         {int(k): sorted(list(v)) for k, v in class_id_to_names_all.items()})

    print(f"[OK] Wrote {len(datasets[tag])} {tag} items to {OUT_ROOT}")

if __name__ == "__main__":
    main()
