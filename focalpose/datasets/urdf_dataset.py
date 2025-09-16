import yaml
import pandas as pd
from pathlib import Path


class Pix3DUrdfDataset:
    def __init__(self, root_dir):
        root_dir = Path(root_dir)
        assert root_dir.exists()

        infos = []
        for category_dir in root_dir.iterdir():
            category = category_dir.name

            for model_dir in category_dir.iterdir():
                urdf_path = model_dir / model_dir.with_suffix('.urdf').name
                info = dict(urdf_path=urdf_path.as_posix(),
                            category=str(category),
                            label=model_dir.name,
                            scale=1)
                infos.append(info)
        self.index = pd.DataFrame(infos)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index.loc[idx]

class CarsUrdfDataset:
    def __init__(self, root_dir):
        root_dir = Path(root_dir)
        assert root_dir.exists()

        infos = []
        for model_dir in root_dir.iterdir():
            urdf_path = model_dir / model_dir.with_suffix('.urdf').name
            info = dict(urdf_path=urdf_path.as_posix(),
                        category='car',
                        label=model_dir.name,
                        scale=1)
            infos.append(info)
        self.index = pd.DataFrame(infos)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index.loc[idx]


class HouseCatDataset:
    def __init__(self, root_dir):
        root_dir = Path(root_dir); assert root_dir.exists()
        infos = []
        for model_dir in sorted(root_dir.iterdir()):
            candidates = list(model_dir.glob("model.urdf")) or list(model_dir.glob("*.urdf"))
            if not candidates: continue
            urdf_path = candidates[0]
            infos.append(dict(
                urdf_path=urdf_path.as_posix(),
                category='hc',
                label=model_dir.name,  # the folder name is the registry label
                scale=1,
            ))
        self.index = pd.DataFrame(infos)

    def __len__(self): return len(self.index)
    def __getitem__(self, idx): return self.index.loc[idx]
