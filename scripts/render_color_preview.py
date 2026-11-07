"""Render full-color multi-angle 3D previews of the Chibi model with Kitsune hair.

Uses pure NumPy + Pillow with material-aware vertex shading, dual-lobe
lighting (diffuse + soft specular highlight), and multi-panel presentation.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh

from chibi_hair import refine_anatomy_sfw


def get_body_color(v_centers: np.ndarray) -> np.ndarray:
    """Compute porcelain skin colors with facial blush, lips, and anatomical accents."""
    n = len(v_centers)
    colors = np.tile([252.0, 228.0, 220.0], (n, 1))  # Porcelain base

    # Cheeks blush (left & right)
    for sign in [-1.0, 1.0]:
        cheek_pos = np.array([sign * 3.2, -6.2, 63.8])
        d = np.linalg.norm(v_centers - cheek_pos, axis=1)
        blush_mask = d < 2.8
        if np.any(blush_mask):
            weight = (np.cos(0.5 * np.pi * (d[blush_mask] / 2.8)) ** 2)[:, None]
            blush_col = np.array([244.0, 172.0, 170.0])
            colors[blush_mask] = colors[blush_mask] * (1 - weight) + blush_col * weight

    # Nose tip soft warmth
    nose_pos = np.array([0.1, -7.29, 63.9])
    d_nose = np.linalg.norm(v_centers - nose_pos, axis=1)
    nose_mask = d_nose < 1.4
    if np.any(nose_mask):
        w = (np.cos(0.5 * np.pi * (d_nose[nose_mask] / 1.4)) ** 2)[:, None]
        colors[nose_mask] = colors[nose_mask] * (1 - w) + np.array([246.0, 185.0, 180.0]) * w

    # Lips
    lip_pos = np.array([0.0, -6.8, 61.2])
    d_lip = np.linalg.norm(v_centers - lip_pos, axis=1)
    lip_mask = d_lip < 1.3
    if np.any(lip_mask):
        w = (np.cos(0.5 * np.pi * (d_lip[lip_mask] / 1.3)) ** 2)[:, None]
        colors[lip_mask] = colors[lip_mask] * (1 - w) + np.array([232.0, 148.0, 154.0]) * w

    # Clavicles / collarbones warmth
    clav_mask = (v_centers[:, 2] >= 53.0) & (v_centers[:, 2] <= 55.5) & (v_centers[:, 1] < -2.0)
    if np.any(clav_mask):
        colors[clav_mask] = colors[clav_mask] * 0.85 + np.array([248.0, 215.0, 208.0]) * 0.15

    # Knees soft warmth
    for sign in [-1.0, 1.0]:
        knee_pos = np.array([sign * 8.5, -4.2, 22.0])
        d_knee = np.linalg.norm(v_centers - knee_pos, axis=1)
        knee_mask = d_knee < 3.5
        if np.any(knee_mask):
            w = (np.cos(0.5 * np.pi * (d_knee[knee_mask] / 3.5)) ** 2)[:, None] * 0.45
            colors[knee_mask] = colors[knee_mask] * (1 - w) + np.array([245.0, 202.0, 195.0]) * w

    return colors


def get_hair_color(v_centers: np.ndarray) -> np.ndarray:
    """Compute anime hair colors: silver-white with lavender tint, pink ear fluff, and dark ties."""
    n = len(v_centers)
    # Base silver-white anime hair with subtle gradient
    z_frac = np.clip((v_centers[:, 2] - 28.0) / (88.0 - 28.0), 0.0, 1.0)[:, None]
    top_col = np.array([246.0, 246.0, 253.0])
    bottom_col = np.array([228.0, 225.0, 240.0])
    colors = top_col * z_frac + bottom_col * (1.0 - z_frac)

    # Inner ear fluff / cavity: soft rosy blossom pink
    for sign in [-1.0, 1.0]:
        ear_pos = np.array([sign * 4.2, 0.2, 78.5])
        d_ear = np.linalg.norm(v_centers - ear_pos, axis=1)
        ear_mask = (d_ear < 3.8) & (v_centers[:, 1] < 1.0)
        if np.any(ear_mask):
            w = (np.cos(0.5 * np.pi * (d_ear[ear_mask] / 3.8)) ** 2)[:, None] * 0.8
            pink_col = np.array([250.0, 190.0, 202.0])
            colors[ear_mask] = colors[ear_mask] * (1 - w) + pink_col * w

    # Hair ties (rings on braids at Z~41.5, Y~-4.6, X~±3.6)
    for sign in [-1.0, 1.0]:
        tie_pos = np.array([sign * 3.6, -4.6, 42.0])
        d_tie = np.linalg.norm(v_centers - tie_pos, axis=1)
        tie_mask = d_tie < 1.1
        if np.any(tie_mask):
            colors[tie_mask] = np.array([92.0, 78.0, 108.0])

    return colors


def render_color_preview(body_path: Path, hair_path: Path, out_path: Path, panel_w=500, panel_h=720) -> None:
    body_mesh = trimesh.load(body_path)
    body_sfw = refine_anatomy_sfw(body_mesh)
    hair_mesh = trimesh.load(hair_path)

    # Cull body triangles that are internal inside the hair dome (back/top of skull)
    # Skull back is Z > 60 and Y > -1.5, which is fully covered by the hair dome
    body_centers = body_sfw.triangles.mean(axis=1)
    internal_skull_mask = (body_centers[:, 2] > 60.0) & (body_centers[:, 1] > -1.5)
    visible_body_tris = body_sfw.triangles[~internal_skull_mask]

    hair_tris = hair_mesh.triangles

    n_body = len(visible_body_tris)
    all_tris = np.vstack([visible_body_tris, hair_tris])
    is_hair = np.zeros(len(all_tris), dtype=bool)
    is_hair[n_body:] = True

    centers = all_tris.mean(axis=1)
    body_colors = get_body_color(centers[:n_body])
    hair_colors = get_hair_color(centers[n_body:])
    all_colors = np.vstack([body_colors, hair_colors])

    # Compute triangle normals
    normals = np.cross(all_tris[:, 1] - all_tris[:, 0], all_tris[:, 2] - all_tris[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms

    views = [
        ("1. FRONT VIEW", "front", None),
        ("2. THREE-QUARTER HERO VIEW", "three_quarter", None),
        ("3. BACK VIEW (FLOWING MANE)", "back", None),
        ("4. SIDE PROFILE", "side", None),
        ("5. BUST CLOSE-UP (KITSUNE EARS & FACE)", "closeup", ((-16.0, -12.0, 46.0), (16.0, 14.0, 89.0))),
    ]

    img = Image.new("RGB", (panel_w * len(views), panel_h + 56), (22, 22, 26))
    draw = ImageDraw.Draw(img)

    for v_idx, (title, mode, crop) in enumerate(views):
        origin_x = v_idx * panel_w

        # Transform vertices into view space
        if mode == "front":
            v_x = all_tris[:, :, 0]
            v_y = all_tris[:, :, 2]
            v_depth = -all_tris[:, :, 1]
            view_normals = normals.copy()
            light_dir = np.array([0.35, -0.75, 0.55])
            fill_dir = np.array([-0.5, -0.4, 0.3])
        elif mode == "back":
            v_x = -all_tris[:, :, 0]
            v_y = all_tris[:, :, 2]
            v_depth = all_tris[:, :, 1]
            view_normals = normals.copy()
            view_normals[:, 0] = -view_normals[:, 0]
            light_dir = np.array([-0.3, 0.8, 0.5])
            fill_dir = np.array([0.4, 0.4, 0.3])
        elif mode == "side":
            v_x = -all_tris[:, :, 1]
            v_y = all_tris[:, :, 2]
            v_depth = all_tris[:, :, 0]
            view_normals = np.column_stack([-normals[:, 1], normals[:, 2], normals[:, 0]])
            light_dir = np.array([-0.4, 0.7, 0.6])
            fill_dir = np.array([0.3, -0.4, 0.2])
        elif mode == "three_quarter":
            ang = np.radians(32.0)
            ca, sa = np.cos(ang), np.sin(ang)
            rot_x = ca * all_tris[:, :, 0] + sa * all_tris[:, :, 1]
            rot_y = -sa * all_tris[:, :, 0] + ca * all_tris[:, :, 1]
            v_x = rot_x
            v_y = all_tris[:, :, 2]
            v_depth = -rot_y
            norm_rx = ca * normals[:, 0] + sa * normals[:, 1]
            norm_ry = -sa * normals[:, 0] + ca * normals[:, 1]
            view_normals = np.column_stack([norm_rx, norm_ry, normals[:, 2]])
            light_dir = np.array([0.4, -0.7, 0.6])
            fill_dir = np.array([-0.5, -0.3, 0.3])
        elif mode == "closeup":
            v_x = all_tris[:, :, 0]
            v_y = all_tris[:, :, 2]
            v_depth = -all_tris[:, :, 1]
            view_normals = normals.copy()
            light_dir = np.array([0.35, -0.75, 0.55])
            fill_dir = np.array([-0.5, -0.4, 0.3])

        light_dir /= np.linalg.norm(light_dir)
        fill_dir /= np.linalg.norm(fill_dir)

        # Depth for sorting: slight bias for hair so outer strands layer cleanly
        depth_vals = v_depth.mean(axis=1) + np.where(is_hair, 0.15, 0.0)
        sort_order = np.argsort(depth_vals)

        # Lighting calculation (Key light + Fill light + Specular)
        dot_key = np.clip(np.sum(view_normals * light_dir, axis=1), 0.0, 1.0)
        dot_fill = np.clip(np.sum(view_normals * fill_dir, axis=1), 0.0, 1.0)

        half_vec = light_dir + np.array([0.0, 0.0, 1.0])
        half_vec /= np.linalg.norm(half_vec)
        dot_spec = np.clip(np.sum(view_normals * half_vec, axis=1), 0.0, 1.0)
        spec = (dot_spec ** 18) * 0.28

        ambient = 0.28
        shade = ambient + 0.62 * dot_key + 0.22 * dot_fill

        if crop:
            mn_x, mx_x = crop[0][0], crop[1][0]
            mn_y, mx_y = crop[0][2], crop[1][2]
        else:
            mn_x, mx_x = v_x.min() - 1.2, v_x.max() + 1.2
            mn_y, mx_y = v_y.min() - 1.2, v_y.max() + 1.2

        scale = min((panel_w - 32) / max(mx_x - mn_x, 1e-6), (panel_h - 32) / max(mx_y - mn_y, 1e-6))
        center_x = (mn_x + mx_x) / 2.0
        center_y = (mn_y + mx_y) / 2.0

        screen_x = origin_x + panel_w / 2 + (v_x - center_x) * scale
        screen_y = panel_h / 2 - (v_y - center_y) * scale

        for idx in sort_order:
            if crop:
                c_orig = centers[idx]
                if not (crop[0][0] <= c_orig[0] <= crop[1][0] and
                        crop[0][1] <= c_orig[1] <= crop[1][1] and
                        crop[0][2] <= c_orig[2] <= crop[1][2]):
                    continue

            sh = shade[idx]
            sp = spec[idx]
            base = all_colors[idx]
            r = int(min(255, base[0] * sh + sp * 255))
            g = int(min(255, base[1] * sh + sp * 255))
            b = int(min(255, base[2] * sh + sp * 255))
            poly = [(screen_x[idx, 0], screen_y[idx, 0]),
                    (screen_x[idx, 1], screen_y[idx, 1]),
                    (screen_x[idx, 2], screen_y[idx, 2])]
            draw.polygon(poly, fill=(r, g, b))

        draw.rectangle([origin_x, panel_h, origin_x + panel_w, panel_h + 56], fill=(16, 16, 20))
        draw.text((origin_x + 12, panel_h + 8), title, fill=(245, 245, 250))
        draw.text((origin_x + 12, panel_h + 28), f"Chibi Pretty v0.0.0.0.0.0-s1 Full Color 3D Preview", fill=(175, 175, 185))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved full-color 3D preview to {out_path} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    body = Path("Chibi_full.stl")
    hair = Path("Chibi_hair.stl")
    out = Path("preview/chibi_s1_color_preview.png")
    render_color_preview(body, hair, out)
