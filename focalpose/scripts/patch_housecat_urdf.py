#!/usr/bin/env python3
from pathlib import Path
import xml.etree.ElementTree as ET

# === EDIT THESE CONSTANTS =====================================================
URDF_ROOT = Path("/mnt/c/Users/whu082/focalpose/local_data/models_urdf/HouseCat")
NORMALIZED_MESH = "model.obj"     # or "meshes/model.obj" if yours live in a subfolder
BACKUP_EXT = ""               # set to "" to disable backups
# ==============================================================================

def patch_urdf(urdf_path: Path) -> bool:
    """Return True if modified."""
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError as e:
        print(f"[PARSE ERR] {urdf_path}: {e}")
        return False

    root = tree.getroot()
    modified = False

    # 1) For every <link>/<visual>/<geometry>/<mesh>, set filename="model.obj" (or your path)
    for link in root.findall(".//link"):
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is not None:
                if mesh.get("filename") != NORMALIZED_MESH:
                    mesh.set("filename", NORMALIZED_MESH)
                    modified = True

    # 2) Remove ALL <collision> blocks anywhere under links
    for link in root.findall(".//link"):
        collisions = link.findall("collision")
        for col in collisions:
            link.remove(col)
            modified = True

    if modified:
        if BACKUP_EXT:
            bak = urdf_path.with_suffix(urdf_path.suffix + BACKUP_EXT)
            bak.write_text(urdf_path.read_text(encoding="utf-8", errors="ignore"),
                           encoding="utf-8")
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return modified

def main():
    print(f"Patching URDFs under: {URDF_ROOT}")
    n_total = n_mod = 0
    for urdf in URDF_ROOT.rglob("*.urdf"):
        n_total += 1
        if patch_urdf(urdf):
            n_mod += 1
            print(f"  [OK]  {urdf}")
        else:
            print(f"  [SKIP] {urdf}")
    print(f"Done. Modified {n_mod}/{n_total} URDFs.")

if __name__ == "__main__":
    main()
