"""Render simple orthographic preview views of a binary STL file.

Pure NumPy + Pillow, no Blender/GPU needed, so it runs anywhere (including a
plain GitHub Actions runner):

    python scripts/render_stl_preview.py Chibi_full.stl preview/Chibi_full_views.png

Coordinate convention of the models in this repository: Z is up and the
figure's front faces -Y.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# title, horizontal axis, vertical axis, depth axis, viewer side (+1/-1),
# optional ((min...), (max...)) crop in world coordinates
VIEWS = [
    ("FRONT  (X-Z, view from -Y)", 0, 2, 1, -1, None),
    ("BACK   (X-Z, view from +Y)", 0, 2, 1, +1, None),
    ("SIDE   (Y-Z, view from +X)", 1, 2, 0, +1, None),
    ("TOP    (X-Y, view from +Z)", 0, 1, 2, +1, None),
    (
        "FRONT CLOSE-UP  (head / neck)",
        0,
        2,
        1,
        -1,
        ((-17.0, -12.0, 47.0), (17.0, 14.0, 81.0)),
    ),
]
BASE_COLOR = (150, 170, 200)


def load_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("STL file is too short")
    count = struct.unpack_from("<I", data, 80)[0]
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    if 84 + count * dtype.itemsize > len(data):
        raise ValueError("Binary STL is truncated")
    records = np.frombuffer(data, dtype=dtype, count=count, offset=84)
    return np.asarray(records["verts"], dtype=np.float64)


def render(triangles: np.ndarray, out_path: Path, panel_w=460, panel_h=700) -> None:
    all_min = triangles.reshape(-1, 3).min(axis=0)
    all_max = triangles.reshape(-1, 3).max(axis=0)
    img = Image.new("RGB", (panel_w * len(VIEWS), panel_h + 52), (24, 24, 28))
    draw = ImageDraw.Draw(img)

    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths[:, None]

    for view_index, (title, ax, ay, depth_axis, side, crop) in enumerate(VIEWS):
        origin_x = view_index * panel_w
        mn = np.array(crop[0]) if crop else all_min - 1.5
        mx = np.array(crop[1]) if crop else all_max + 1.5
        scale = min(
            (panel_w - 24) / max(mx[ax] - mn[ax], 1e-9),
            (panel_h - 24) / max(mx[ay] - mn[ay], 1e-9),
        )
        center = (mn + mx) / 2.0

        depth = triangles[:, :, depth_axis].mean(axis=1)
        # painter's algorithm: far triangles first
        order = np.argsort(depth if side > 0 else -depth)
        light = np.zeros(3)
        light[depth_axis] = side
        light[ay] += 0.45
        light /= np.linalg.norm(light)
        shade = np.abs(normals @ light) * 0.75 + 0.25

        xs = origin_x + panel_w / 2 + (triangles[:, :, ax] - center[ax]) * scale
        ys = panel_h / 2 - (triangles[:, :, ay] - center[ay]) * scale
        for i in order:
            color = tuple(int(min(255, c * shade[i])) for c in BASE_COLOR)
            draw.polygon(
                [(xs[i, 0], ys[i, 0]), (xs[i, 1], ys[i, 1]), (xs[i, 2], ys[i, 2])],
                fill=color,
            )

        draw.rectangle(
            [origin_x, panel_h, origin_x + panel_w, panel_h + 52], fill=(16, 16, 20)
        )
        draw.text((origin_x + 10, panel_h + 7), title, fill=(240, 240, 240))
        draw.text(
            (origin_x + 10, panel_h + 25),
            f"{'XYZ'[ax]}: {mn[ax]:.1f}..{mx[ax]:.1f}   "
            f"{'XYZ'[ay]}: {mn[ay]:.1f}..{mx[ay]:.1f}   tris: {len(triangles)}",
            fill=(170, 170, 180),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"wrote {out_path} ({img.size[0]}x{img.size[1]})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stl")
    parser.add_argument("png")
    args = parser.parse_args(argv)
    triangles = load_stl_triangles(Path(args.stl))
    print(
        f"{args.stl}: {len(triangles)} triangles, "
        f"bbox min {triangles.reshape(-1, 3).min(axis=0)}, "
        f"max {triangles.reshape(-1, 3).max(axis=0)}"
    )
    render(triangles, Path(args.png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
