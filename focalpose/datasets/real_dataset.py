from difflib import get_close_matches
import re
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R
import json

from .utils import make_masks_from_det
from .datasets_cfg import make_urdf_dataset


class Pix3DDataset:
    def __init__(self, ds_dir, category, train=True):
        self.ds_dir = Path(ds_dir)
        self.train = train
        assert self.ds_dir.exists()
        df = pd.read_json((self.ds_dir / 'pix3d.json').as_posix())
        mask = (df['category'] == category) & (df['occluded'] == False) & (df['truncated'] == False) & (
                    df['slightly_occluded'] == False)
        index = df[mask].reset_index(drop=True)

        test_list = self.ds_dir / 'pix3d_test_list.txt'

        test_ids = []
        with test_list.open() as f:
            for i in f:
                test_ids.append('img' + i.replace('\n', ''))

        # Drop images that break our pipeline (1 in each class)
        if category == 'table':
            index = index.drop(index=257)
        elif category == 'sofa':
            index = index.drop(index=76)
        elif category == 'chair':
            index = index.drop(index=2563)
        elif category == 'bed':
            index = index.drop(index=217)
        self.index = index.reset_index(drop=True)

        if self.train:
            mask = ~self.index['img'].isin(test_ids)
        else:
            mask = self.index['img'].isin(test_ids)

        self.index = self.index[mask].reset_index(drop=True)

        if category == 'chair':
            # Fix multiple models in one category
            multiple_models = ['IKEA_JULES_1',
                               'IKEA_MARKUS',
                               'IKEA_PATRIK',
                               'IKEA_SKRUVSTA',
                               'IKEA_SNILLE_1']

            for model in multiple_models:
                mask = self.index['model'].str.contains(model)
                self.index.loc[mask, 'model'] = f'model/chair/{model}/model.obj'

        self.category = category
        self.R_pix3d = R.from_euler('xyz', np.pi * np.array([0, 0, 1]))
        urdf_ds = make_urdf_dataset(f'{ds_dir.as_posix().split("/")[-1]}-{category}')
        self.all_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        focal_length = entry['focal_length']
        resolution = entry['img_size']
        focal_length = (focal_length * resolution[0]) / 32
        K = np.array([[focal_length, 0, resolution[0] / 2], [0, focal_length, resolution[1] / 2], [0, 0, 1]])
        rgb = np.asarray(Image.open((self.ds_dir / entry['img'])).convert('RGB'))
        mask = np.array(np.asarray(Image.open((self.ds_dir / entry['mask']))) / 255, dtype=np.uint8)

        t = self.R_pix3d.apply(entry['trans_mat']).reshape(3, -1)
        R = self.R_pix3d.as_matrix() @ np.array(entry['rot_mat'])
        TWC = np.linalg.inv(np.vstack([np.hstack([R, t]), [0, 0, 0, 1]]))
        name = entry['model'].replace(f'model/{self.category}/', '').replace('/model.obj',
                                                                             '') + f'_{self.category.upper()}'
        camera = dict(TWC=TWC, K=K, resolution=resolution)
        objects = dict(TWO=np.eye(4), name=name, scale=1, id_in_segm=1, bbox=np.array(entry['bbox']))

        return rgb, mask, dict(camera=camera, objects=[objects])


class StanfordCars3DDataset:
    def __init__(self, ds_dir, train=True):
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists()
        self.train = train
        if self.train:
            self.index = pd.read_pickle((self.ds_dir / 'train_anno_preprocessed.pkl').as_posix())
        else:
            self.index = pd.read_pickle((self.ds_dir / 'test_anno_preprocessed.pkl').as_posix())
        urdf_ds = make_urdf_dataset('stanfordcars3d')
        self.all_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]

    @staticmethod
    def resize_rgb(rgb):
        new_width = 300
        new_height = int(new_width * rgb.size[1] / rgb.size[0])
        return np.asarray(rgb.resize((new_width, new_height), Image.ANTIALIAS))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        K = entry['K']
        TCO = entry['TCO']
        bbox = entry['bbox']
        name = entry['model_id']

        if self.train:
            rgb = self.resize_rgb(Image.open((self.ds_dir / 'cars_train' / entry['img'])).convert('RGB'))
        else:
            rgb = self.resize_rgb(Image.open((self.ds_dir / 'cars_test' / entry['img'])).convert('RGB'))

        mask = make_masks_from_det([np.clip(bbox, a_min=0, a_max=None)], rgb.shape[0], rgb.shape[1]).squeeze(0).numpy()

        camera = dict(TWC=np.linalg.inv(TCO), K=K, resolution=rgb.shape[:2])
        objects = dict(TWO=np.eye(4), name=name, scale=1, id_in_segm=1, bbox=bbox)

        return rgb, mask, dict(camera=camera, objects=[objects])


class CompCars3DDataset:
    def __init__(self, ds_dir, train=True):
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists()
        self.train = train
        if self.train:
            self.index = pd.read_pickle((self.ds_dir / 'train_anno_preprocessed.pkl').as_posix())
        else:
            self.index = pd.read_pickle((self.ds_dir / 'test_anno_preprocessed.pkl').as_posix())
        urdf_ds = make_urdf_dataset('compcars3d')
        self.all_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]

    @staticmethod
    def resize_rgb(rgb):
        new_width = 300
        new_height = int(new_width * rgb.size[1] / rgb.size[0])
        return np.asarray(rgb.resize((new_width, new_height), Image.ANTIALIAS))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        K = entry['K']
        TCO = entry['TCO']
        bbox = entry['bbox']
        name = entry['model_id']

        rgb = self.resize_rgb(Image.open((self.ds_dir / 'data' / 'image' / entry['img'])).convert('RGB'))
        mask = make_masks_from_det([np.clip(bbox, a_min=0, a_max=None)], rgb.shape[0], rgb.shape[1]).squeeze(0).numpy()

        camera = dict(TWC=np.linalg.inv(TCO), K=K, resolution=rgb.shape[:2])
        objects = dict(TWO=np.eye(4), name=name, scale=1, id_in_segm=1, bbox=bbox)

        return rgb, mask, dict(camera=camera, objects=[objects])


def _K_33_from_fx_fy_cx_cy(K_list):
    fx, fy, cx, cy = map(float, K_list)
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def _TCO_from_R_t(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, dtype=np.float32)
    T[:3, 3]  = np.asarray(t, dtype=np.float32).reshape(3,)
    return T

def _read_json_rows(json_path: Path):
    rows = []
    with open(json_path, "r") as f:
        for rec in json.load(f):
            K = _K_33_from_fx_fy_cx_cy(rec['K'])
            img_rel = rec['image_path']
            for obj in rec.get('objects', []):
                rows.append(dict(
                    img=img_rel,
                    K=K,
                    TCO=_TCO_from_R_t(obj['R'], obj['t']),
                    bbox=np.asarray(obj['bbox_xywh'], dtype=np.float32),
                    model_id=obj.get('name', ''),      # full name from meta.txt
                    model_path=obj.get('model', ''),   # optional absolute .obj
                    scene_id=rec.get('scene_id', ''),
                    frame_id=rec.get('frame_id', '')
                ))
    return pd.DataFrame(rows)

class HouseCatDataset:
    """
    Reads annotations/housecat_{split}.json and returns one object per sample.
    Resolves per-instance names to a valid URDF label using both the URDF registry
    and a fallback scan of models_urdf/HouseCat if needed. Drops unresolved rows.
    """
    def __init__(self, ds_dir, train=True, split=None, strict_urdf=True,
                 urdf_name='housecat', urdf_dir=None):
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists(), f"Missing dataset root: {self.ds_dir}"

        self.split = ('train' if train else 'test') if split is None else split
        json_path = self.ds_dir / 'annotations' / f'housecat_{self.split}.json'
        assert json_path.exists(), f"Missing {json_path}"

        # --- Load rows from JSON
        self.index = _read_json_rows(json_path)

        # --- Build the set of available labels (union of URDF registry + folder scan)
        urdf_labels = []
        try:
            urdf_ds = make_urdf_dataset(urdf_name)
            urdf_labels = [obj['label'] for _, obj in urdf_ds.index.iterrows()]
        except Exception:
            pass

        # Fallback: scan models_urdf/HouseCat alongside your project, or use user-provided dir
        scan_dirs = []
        if urdf_dir is not None:
            scan_dirs.append(Path(urdf_dir))
        # common locations relative to ds_dir
        scan_dirs += [
            (self.ds_dir.parent / "models_urdf" / "HouseCat"),
            (self.ds_dir.parent / "models_urdf" / "housecat"),
            (self.ds_dir.parent.parent / "models_urdf" / "HouseCat"),
            (self.ds_dir.parent.parent / "models_urdf" / "housecat"),
        ]
        scanned = []
        for d in scan_dirs:
            if d.exists():
                scanned += [p.name for p in d.iterdir() if p.is_dir()]
        # merge & canonicalize (preserve original casing from urdf_labels first)
        urdf_labels = list(dict.fromkeys(urdf_labels + scanned))

        # If still empty, fail early with a helpful message
        assert len(urdf_labels) > 0, (
            "No URDF labels found. Either your 'housecat' URDF pack is not registered, "
            "or 'urdf_dir' doesn't point to models_urdf/HouseCat. "
            "Pass urdf_dir=... to HouseCatDataset."
        )

        self._urdf_labels = urdf_labels
        self._urdf_lc = [u.lower() for u in urdf_labels]
        self._urdf_set = set(self._urdf_lc)
        self._lower2canon = {u.lower(): u for u in urdf_labels}

        # --- Resolve names to URDF labels
        self.index['mesh_label'] = self.index['model_id'].apply(self._resolve_label)

        # --- Show a few mappings for sanity
        shown = 0
        print("[INFO] Example label resolutions:")
        for orig, resolved in zip(self.index['model_id'].tolist(), self.index['mesh_label'].tolist()):
            if isinstance(orig, str) and isinstance(resolved, str) and orig != resolved:
                print(f"  {orig} -> {resolved}")
                shown += 1
                if shown >= 8:
                    break
        if shown == 0:
            print("  (no changes; dataset names already match URDF labels)")

        # --- Drop unresolved labels to prevent KeyError later
        keep = self.index['mesh_label'].str.lower().isin(self._urdf_set)
        dropped = int((~keep).sum())
        if dropped:
            bad = self.index.loc[~keep, 'model_id'].unique().tolist()[:10]
            print(f"[WARN] Dropping {dropped} items with unknown URDF labels (e.g., {bad})")
        self.index = self.index[keep].reset_index(drop=True)

        # also keep canonical label list for quick checks
        self.all_labels = urdf_labels

    # ---------- resolution logic ----------
    def _tokenize(self, s: str):
        return [t for t in re.split(r'[_\-\s]+', s.lower()) if t]

    def _resolve_label(self, s: str) -> str:
        """Map dataset name to a URDF label using multiple strategies."""
        if not isinstance(s, str) or not s:
            return s
        sl = s.strip().lower()

        # 0) exact
        if sl in self._lower2canon:
            return self._lower2canon[sl]

        # 1) dash/underscore variants
        variants = (sl, sl.replace('-', '_'), sl.replace('_', '-'))
        for v in variants:
            if v in self._lower2canon:
                return self._lower2canon[v]

        # 2) prefix completion (e.g., 'shoe-sky' -> 'shoe-sky_blue_holes_right')
        pref = [u for u in self._urdf_labels if u.lower().startswith(sl + '_') or u.lower().startswith(sl + '-')]
        if len(pref) == 1:
            return pref[0]
        if len(pref) > 1:
            return min(pref, key=len)

        # 3) substring anywhere
        sub = [u for u in self._urdf_labels if sl in u.lower()]
        if len(sub) == 1:
            return sub[0]
        if len(sub) > 1:
            return min(sub, key=len)

        # 4) token overlap score
        stoks = set(self._tokenize(sl))
        if stoks:
            best, best_score = None, 0.0
            for u in self._urdf_labels:
                utoks = set(self._tokenize(u))
                if not utoks:
                    continue
                score = len(stoks & utoks) / float(len(stoks))
                if score > best_score:
                    best, best_score = u, score
            if best is not None and best_score >= 0.6:
                return best

        m = get_close_matches(sl, self._urdf_lc, n=1, cutoff=0.8)
        if m:
            return self._lower2canon[m[0]]

        # give up — will be filtered out by 'keep' mask
        return s

    # ---------- dataset API ----------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        e = self.index.iloc[idx]
        rgb = np.asarray(Image.open((self.ds_dir / e['img'])).convert('RGB'))
        H, W = rgb.shape[:2]

        # enforce float32 everywhere
        K   = np.asarray(e['K'],   np.float32)
        TCO = np.asarray(e['TCO'], np.float32)
        TWC = np.linalg.inv(TCO).astype(np.float32, copy=False)
        TWO = np.eye(4, dtype=np.float32)
        bbox = np.asarray(e['bbox'], np.float32)

        mask = make_masks_from_det([np.clip(bbox, a_min=0, a_max=None)], H, W).squeeze(0).numpy()

        # use resolved label so mesh_db.select(labels) always succeeds
        name = e['mesh_label']

        camera = dict(TWC=TWC, K=K, resolution=rgb.shape[:2])
        objects = dict(TWO=TWO, name=name, scale=1, id_in_segm=1, bbox=bbox)

        if isinstance(e.get('model_path'), str) and e['model_path']:
            objects['mesh_path'] = e['model_path']

        return rgb, mask, dict(camera=camera, objects=[objects])