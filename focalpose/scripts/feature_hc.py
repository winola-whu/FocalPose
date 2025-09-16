#!/usr/bin/env python3
"""
HouseCat DINO Feature Builder (separate from the refiner)
--------------------------------------------------------
Builds cached DINO ViT-B/8 features exactly like single-image inference, using
your annotations JSON and the per-object bboxes.

Output files (written to FEATURES_DIR):
  - <ds_key>_train_features.npy
  - <ds_key>_train_y.npy
  - (optional) <ds_key>_test_features.npy, <ds_key>_test_y.npy

Usage example:
  python -m focalpose.scripts.housecat_build_features \
      --images-root /mnt/c/Users/whu082/focalpose/local_data \
      --ann-train   /mnt/c/Users/whu082/focalpose/local_data/annotations/train.json \
      --ann-test    /mnt/c/Users/whu082/focalpose/local_data/annotations/test.json \
      --ds-key housecat

Then run your refiner normally:
  python -m focalpose.scripts.HousecatRefiner

Notes:
- This script mirrors the ROIAlign(224x224) → ImageNet normalize → ViT-B/8 (DINO) → L2-normalize pipeline.
- Labels default to each object's `name` string from the annotation.
"""

import argparse, os
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torchvision import transforms as pth_transforms
import torchvision
from tqdm import tqdm

from focalpose.config import LOCAL_DATA_DIR, FEATURES_DIR
from focalpose.models import vision_transformer as vits

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------
# DINO feature extractor (ViT-B/8) – mirrors single-image inference
# ----------------------------

def build_dino_feature_extractor(device: torch.device):
    vit_model = vits.__dict__["vit_base"](patch_size=8, num_classes=0)
    for p in vit_model.parameters():
        p.requires_grad = False
    vit_model.eval().to(device)

    ckpt = (LOCAL_DATA_DIR / "dino_vitbase8_pretrain.pth").as_posix()
    state = torch.load(ckpt, map_location=device)
    vit_model.load_state_dict(state, strict=True)
    return vit_model


# Exact ImageNet normalization
_IMAGENET_NORM = pth_transforms.Normalize(
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
)


@torch.no_grad()
def dino_feature_from_bbox(vit_model, rgb_bchw: torch.Tensor, bbox_xyxy, device: torch.device) -> torch.Tensor:
    """Return a L2-normalized DINO feature [D] from an image tensor and a bbox.
    - rgb_bchw: [1,3,H,W] in [0,1]
    - bbox_xyxy: [x1,y1,x2,y2] in pixels
    """
    if not isinstance(bbox_xyxy, torch.Tensor):
        bbox_xyxy = torch.tensor(bbox_xyxy, dtype=torch.float32)
    roi = bbox_xyxy.unsqueeze(0).to(device)
    crop = torchvision.ops.roi_align(rgb_bchw.to(device), [roi], output_size=(224, 224))[0]
    crop = _IMAGENET_NORM(crop)
    feat = vit_model(crop.unsqueeze(0))  # [1,D]
    feat = F.normalize(feat, dim=-1)[0]
    return feat


# ----------------------------
# Annotation-driven feature building
# ----------------------------

class FeatureBuilder:
    def __init__(self, ds_key: str = "housecat", label_field: str = "name"):
        self.ds_key = ds_key
        self.label_field = label_field

    @staticmethod
    def _xywh_to_xyxy(b):
        x1, y1, x2, y2 = b
        # If your format is (x,y,w,h), change to: x2=x1+w, y2=y1+h
        return float(x1), float(y1), float(x2), float(y2)

    def _load_image_tensor(self, path: Path, device: torch.device) -> Tuple[np.ndarray, torch.Tensor]:
        arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return arr, ten

    @torch.no_grad()
    def build_split(self, vit_model, device, images_root: Path, ann_path: Path, split: str):
        images_root = images_root.resolve()
        with open(ann_path, "r") as f:
            records = json.load(f)

        # Count total objects for a precise progress bar
        total_objs = 0
        for rec in records:
            total_objs += len(rec.get("objects", []))

        X_list: List[np.ndarray] = []
        y_list: List[str] = []

        pbar = tqdm(total=total_objs if total_objs > 0 else None,
                    desc=f"{split}: objects", unit="obj")
        for rec in records:
            img_rel = rec.get("image_path")
            if not img_rel:
                continue
            img_path = images_root / img_rel
            if not img_path.exists():
                # fallback: join with split folder
                img_path = images_root / "images" / split / Path(img_rel).name
            if not img_path.exists():
                log(f"[warn] missing image: {img_rel}")
                # still advance bar by number of objects to keep counts aligned
                pbar.update(len(rec.get("objects", [])))
                continue

            arr, t_im = self._load_image_tensor(img_path, device)
            H, W = arr.shape[:2]

            for obj in rec.get("objects", []):
                label = str(obj.get(self.label_field, "unknown"))
                bbox = obj.get("bbox_xywh")
                if bbox is None:
                    pbar.update(1)
                    continue
                x1, y1, x2, y2 = self._xywh_to_xyxy(bbox)
                # clamp to image bounds and ensure non-empty
                x1 = max(0.0, min(x1, W - 1))
                y1 = max(0.0, min(y1, H - 1))
                x2 = max(x1 + 1.0, min(x2, W))
                y2 = max(y1 + 1.0, min(y2, H))

                feat = dino_feature_from_bbox(vit_model, t_im, [x1, y1, x2, y2], device)
                X_list.append(feat.detach().cpu().numpy())
                y_list.append(label)
                pbar.update(1)
        pbar.close()

        if not X_list:
            raise RuntimeError(f"No features built from {ann_path}. Check paths and annotation format.")

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list)
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        np.save(FEATURES_DIR / f"{self.ds_key}_{split}_features.npy", X)
        np.save(FEATURES_DIR / f"{self.ds_key}_{split}_y.npy", y)
        log(f"Saved {split} features: X={X.shape}  y={y.shape}  → {FEATURES_DIR}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=Path, default="local_data/HouseCat")
    ap.add_argument("--ann-train", type=Path, default="local_data/HouseCat/annotations/housecat_train.json")
    ap.add_argument("--ann-test", type=Path, default="local_data/HouseCat/annotations/housecat_test.json",
                    help="Optional: test annotations JSON; will write *_test_*.npy as well.")
    ap.add_argument("--ds-key", type=str, default="housecat",
                    help="Prefix for saved feature files in FEATURES_DIR.")
    ap.add_argument("--label-field", type=str, default="name",
                    help="Field in each object to use as label (default: 'name').")
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log("Loading DINO ViT-B/8…")
    dino = build_dino_feature_extractor(device)

    fb = FeatureBuilder(ds_key=args.ds_key, label_field=args.label_field)
    log("Building TRAIN features…")
    fb.build_split(dino, device, args.images_root, args.ann_train, split="train")

    if args.ann_test is not None:
        log("Building TEST features…")
        fb.build_split(dino, device, args.images_root, args.ann_test, split="test")

    log("Done.")


if __name__ == "__main__":
    main()
