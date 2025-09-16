#!/usr/bin/env python3
# - DINO features computed:
#   * ViT-B/8 from focalpose.models.vision_transformer (no torch.hub)
#   * ROIAlign crop to 224x224 (full-image box by default), ImageNet normalize
#   * eval-mode, frozen weights; checkpoint at LOCAL_DATA_DIR/dino_vitbase8_pretrain.pth


import argparse, json, os, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# FocalPose imports (keep aligned with run_single_image_inference)
from focalpose.config import EXP_DIR, LOCAL_DATA_DIR, FEATURES_DIR
from focalpose.training.pose_models_cfg import create_model_pose
from focalpose.rendering.bullet_batch_renderer import BulletBatchRenderer
from focalpose.datasets.datasets_cfg import make_urdf_dataset
from focalpose.lib3d.rigid_mesh_database import MeshDataBase
from focalpose.models import vision_transformer as vits

# torchvision utils for ROIAlign + normalization
import torchvision
from torchvision import transforms as pth_transforms

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------
# CLI (keep your defaults)
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, default="imgs/B1_cropped.jpg")
    ap.add_argument("--mesh-root", type=Path, default="local_data/models_urdf")
    ap.add_argument("--refine-run-id", type=str, default="housecat-refine-huber-F10--137946")
    ap.add_argument("--R", nargs=9, type=float,
                    default=[0.98564703, 0.0096622, 0.16854252, 0.02498538, -0.99571518, -0.08903371, 0.16696009,
                             0.09196691, -0.98166512])
    ap.add_argument("--t", nargs=3, type=float, default=[0.0113753, 0.05409772, 0.25418229])
    ap.add_argument("--f", type=float, default=975)

    # Small knobs (unchanged)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


# ----------------------------
# Utilities
# ----------------------------

def intrinsics_center(f, W, H):
    cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)


def load_image_unchanged(path: Path, device: torch.device):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return arr, ten


def enumerate_housecat_instances(mesh_root: Path) -> Dict[str, str]:
    """
    Expect mesh_root/HouseCat/<instance_name>/model.urdf
    Returns: {instance_name: absolute_model_path}
    """
    out = {}
    hc = mesh_root / "HouseCat"
    if not hc.exists():
        raise FileNotFoundError(f"Expected {hc} to exist")
    for inst_dir in hc.iterdir():
        if not inst_dir.is_dir():
            continue
        urdf = inst_dir / "model.urdf"
        if urdf.exists():
            out[inst_dir.name] = str(urdf.resolve())
    if not out:
        raise RuntimeError(f"No instances found under {hc}")
    return out


# ----------------------------
# DINO feature extractor (ViT-B/8) – mirror single-image inference
# ----------------------------


def build_dino_feature_extractor(device: torch.device):
    # ViT-B/8 headless (num_classes=0). Weights from LOCAL_DATA_DIR
    vit_model = vits.__dict__["vit_base"](patch_size=8, num_classes=0)
    for p in vit_model.parameters():
        p.requires_grad = False
    vit_model.eval().to(device)

    ckpt = (LOCAL_DATA_DIR / "dino_vitbase8_pretrain.pth").as_posix()
    state = torch.load(ckpt, map_location=device)
    vit_model.load_state_dict(state, strict=True)
    return vit_model


# Exact normalize used by the runner (ImageNet stats)
_imagenet_normalize = pth_transforms.Normalize(
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
)


@torch.no_grad()
def dino_feature_from_bbox(vit_model, rgb_bchw: torch.Tensor, bbox_xyxy, device: torch.device) -> torch.Tensor:
    """
    rgb_bchw: [1,3,H,W] in [0,1]
    bbox_xyxy: tensor/list [x1,y1,x2,y2] in pixel coords (same scale as rgb)
    Returns: L2-normalized feature [D]
    """
    if not isinstance(bbox_xyxy, torch.Tensor):
        bbox_xyxy = torch.tensor(bbox_xyxy, dtype=torch.float32)
    box = bbox_xyxy.unsqueeze(0).to(device)
    # ROIAlign to 224×224 with the same path used by the reference runner
    crop = torchvision.ops.roi_align(rgb_bchw.to(device), [box], output_size=(224, 224))[0]
    crop = _imagenet_normalize(crop)
    feat = vit_model(crop.unsqueeze(0))  # [1, D]
    feat = F.normalize(feat, dim=-1)[0]
    return feat


# ----------------------------
# Classifier reuse via cached dataset features
# ----------------------------

class InstanceRanker:
    """Ranks instance labels using cached train features from FEATURES_DIR.
    Strategy: L2-normalize all features and use nearest-centroid (mean feature per label).
    This mirrors using dataset features (fast, deterministic) without changing CLI.
    """

    def __init__(self, ds_key: str = "housecat"):
        self.ds_key = ds_key
        self.centroids: Dict[str, np.ndarray] = {}

    def load_cached_features(self):
        # Expected files (as produced by the single-image inference preparation):
        #   <ds_key>_train_features.npy, <ds_key>_train_y.npy
        Xp = FEATURES_DIR / f"{self.ds_key}_train_features.npy"
        Yp = FEATURES_DIR / f"{self.ds_key}_train_y.npy"
        if not Xp.exists() or not Yp.exists():
            raise FileNotFoundError(
                f"Cached features not found. Expected: {Xp} and {Yp}. "
                f"Please generate train features for '{self.ds_key}'."
            )
        X = np.load(Xp)
        y = np.load(Yp, allow_pickle=True)
        # L2-normalize
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        # Build per-label centroid
        by_label: Dict[str, List[np.ndarray]] = {}
        for xi, yi in zip(X, y):
            by_label.setdefault(str(yi), []).append(xi)
        self.centroids = {lab: np.mean(np.stack(v, 0), 0) for lab, v in by_label.items()}
        # Normalize centroids
        for k in list(self.centroids.keys()):
            c = self.centroids[k]
            self.centroids[k] = c / (np.linalg.norm(c) + 1e-10)
        log(f"Loaded cached train features for {len(self.centroids)} labels from {FEATURES_DIR}")

    def rank(self, qfeat: np.ndarray, topk: int) -> List[Tuple[str, float]]:
        if not self.centroids:
            self.load_cached_features()
        # Cosine similarity to centroids (features already L2-normalized)
        sims = []
        for lab, c in self.centroids.items():
            sims.append((lab, float(np.dot(qfeat, c))))
        sims.sort(key=lambda x: -x[1])
        return sims[:topk]


# ----------------------------
# Refiner loading (unchanged)
# ----------------------------

def load_pose_model(run_id, cfg, batch_renderer, mesh_db, device):
    log(f"Creating refiner with create_model_pose(...) for run '{run_id}'")
    model = create_model_pose(cfg=cfg, renderer=batch_renderer, mesh_db=mesh_db).to(device).float()
    pth_dir = EXP_DIR / run_id
    path = pth_dir / 'checkpoint.pth.tar'
    save = torch.load(path.as_posix(), map_location=device)
    state_dict = save['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ----------------------------
# Main
# ----------------------------

def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    outdir = Path("outputs/housecat_full")
    outdir.mkdir(parents=True, exist_ok=True)

    # 0) Minimal cfg like the official runner
    cfg = argparse.ArgumentParser('')  # dummy parser to hold attrs
    cfg = cfg.parse_args([])
    cfg.backbone_str = 'resnet50'
    cfg.backbone_pretrained = True
    cfg.n_pose_dims = 9
    cfg.n_dataloader_workers = 8
    cfg.niter = 12
    cfg.img = str(a.img)
    cfg.cls = 'housecat'
    cfg.topk = a.topk
    cfg.input_resize = (640, 640)
    cfg.n_rendering_workers = 4

    # 1) Load image (NO resize) + intrinsics + init pose
    rgb_np, rgb_t = load_image_unchanged(a.img, device)
    H, W = rgb_np.shape[:2]
    log(f"Image loaded: {W}x{H}")
    if not (H == 640 and W == 640):
        print(f"[warn] input image is {W}x{H}; script assumes 640x640 but continues unchanged.")
    K = intrinsics_center(a.f, W, H)
    R = np.asarray(a.R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(a.t, dtype=np.float32).reshape(3, 1)

    # 2) Enumerate HouseCat instances
    log("Enumerating HouseCat instances…")
    instances = enumerate_housecat_instances(a.mesh_root)
    labels = list(instances.keys())
    log(f"Found {len(labels)} instances under {a.mesh_root}/HouseCat")

    # 3) DINO model (ViT-B/8) + query feature via ROIAlign( full-image box )
    log("Loading DINO ViT-B/8 and computing query feature…")
    dino = build_dino_feature_extractor(device)
    full_box = torch.tensor([0.0, 0.0, float(W), float(H)], dtype=torch.float32)
    with torch.no_grad():
        qfeat_t = dino_feature_from_bbox(dino, rgb_t, full_box, device)
        qfeat = qfeat_t.detach().cpu().numpy()

    # 4) Rank instances using cached dataset features (nearest-centroid)
    log("Ranking instances via cached train features (nearest-centroid)…")
    ranker = InstanceRanker(ds_key='housecat')
    ranking = ranker.rank(qfeat, topk=a.topk)
    for r, (iid, sc) in enumerate(ranking, 1):
        log(f"  #{r}: {iid}  cos={sc:.4f}")

    # 5) Prepare MeshDB/BatchRenderer and load the refiner
    log("Preparing MeshDB/BatchRenderer and loading refiner…")
    urdf_name = "housecat"
    urdf_ds = make_urdf_dataset(urdf_name)
    mesh_db = MeshDataBase.from_urdf_ds(urdf_ds).batched().to(device).float()
    batch_renderer = BulletBatchRenderer(
        object_set=urdf_name,
        n_workers=cfg.n_rendering_workers,
        preload_cache=False,
        split_objects=True,
    )
    model = load_pose_model(a.refine_run_id, cfg, batch_renderer, mesh_db, device)

    # 6) Refine each top-k candidate (same forward signature)
    n_iter = 8
    summary = []
    log(f"Starting refinement for top-{len(ranking)} (n_iter={n_iter})…")
    for rank, (iid, score) in enumerate(ranking, 1):
        # Build TCO_init from your R,t
        TCO_init_np = np.eye(4, dtype=np.float32)
        TCO_init_np[:3, :3] = R
        TCO_init_np[:3, 3] = t.reshape(3)
        TCO_init = torch.from_numpy(TCO_init_np)[None].to(device)  # [1,4,4]

        images = rgb_t.float()  # [1,3,H,W] in [0,1]
        K_t = torch.from_numpy(K)[None].float().to(device)  # [1,3,3]

        with torch.no_grad():
            outputs = model(
                images=images,
                K=K_t,
                labels=[iid],
                TCO=TCO_init,
                n_iterations=n_iter,
                update_focal_length=True,
            )

        iter_outputs = outputs[f"iteration={n_iter}"]
        TCO_pred = iter_outputs["TCO_output"]  # [1,4,4]
        K_pred = iter_outputs["K_output"]  # [1,3,3]

        TCO_pred_np = TCO_pred[0].detach().cpu().numpy()
        R_out = TCO_pred_np[:3, :3]
        t_out = TCO_pred_np[:3, 3:4]
        K_ref = K_pred[0].detach().cpu().numpy()
        f_scalar = float((K_ref[0, 0] + K_ref[1, 1]) * 0.5)  # fx≈fy

        rdir = Path(outdir) / f"rank_{rank:02d}__{iid}"
        rdir.mkdir(parents=True, exist_ok=True)
        np.savetxt(rdir / "K_refined.txt", K_ref, fmt="%.6f")
        np.savetxt(rdir / "R_refined.txt", R_out, fmt="%.6f")
        np.savetxt(rdir / "t_refined.txt", t_out, fmt="%.6f")
        with open(rdir / "result.json", "w") as f:
            json.dump({
                "instance_id": iid,
                "retrieval_score_cosine": score,
                "iters": n_iter,
                "K_init": K.tolist(), "K_refined": K_ref.tolist(),
                "R_init": R.tolist(), "R_refined": R_out.tolist(),
                "t_init": t.reshape(3).tolist(), "t_refined": t_out.reshape(3).tolist(),
                "f_init": float(a.f), "f_refined": f_scalar,
            }, f, indent=2)
        summary.append({"rank": rank, "instance_id": iid, "score": score, "dir": str(rdir)})

    log("[refine] done")
    with open(Path(outdir) / "summary_topk.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[ok] wrote:", outdir)
    for row in summary:
        print(f"  #{row['rank']:02d} {row['instance_id']} cos={row['score']:.3f} -> {row['dir']}")


if __name__ == "__main__":
    main()
