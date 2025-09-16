#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_ROOT = Path("/mnt/c/Users/whu082/focalpose/local_data/models_urdf/HouseCat")

def find_local_obj(urdf_path: Path):
    folder = urdf_path.parent
    base = folder.name  # e.g., teapot-white_rectangle_sprout
    # Prefer <label>.obj, then model.obj, then any *.obj
    for name in [f"{base}.obj", "model.obj"]:
        p = folder / name
        if p.exists():
            return name
    any_obj = sorted(folder.glob("*.obj"))
    return any_obj[0].name if any_obj else None

fixed = 0
for urdf in URDF_ROOT.rglob("*.urdf"):
    try:
        tree = ET.parse(urdf)
    except ET.ParseError:
        continue
    root = tree.getroot()
    obj_name = find_local_obj(urdf)
    if not obj_name:
        continue
    changed = False
    for mesh in root.findall(".//mesh"):
        fn = mesh.get("filename", "")
        if fn != obj_name:
            mesh.set("filename", obj_name)
            changed = True
    if changed:
        tree.write(urdf, encoding="utf-8", xml_declaration=True)
        fixed += 1

print(f"[OK] Rewrote {fixed} URDFs to reference OBJ in the same folder.")
