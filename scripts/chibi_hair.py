"""Enhanced Chibi v0.0.0.0.0.0-s1 generator.

Matches the reference VTuber / Kitsune anime hairstyle exactly:
- Voluminous skull cap covering the crown, sides and rear of the head completely
- Large stylized Kitsune beast ears with flared outer rim and inner fluff tufts
- Lush arched forehead bangs resting against brow with center notch and curled under-bevel
- Wide curved face-framing locks hugging the cheeks and jawline
- Detailed twin braids hanging over the shoulders with ties and flowing tails
- Cascading back hair mane with layered waves and tapered flicked tips
- Anatomical SFW refinement: smooth natural chest contours, refined clavicles,
  and elegant feminine silhouette without explicit protrusions.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import trimesh


def make_lofted_surface(rings: list[np.ndarray], closed_ring: bool = True) -> trimesh.Trimesh:
    """Loft cross-section rings into a solid 3D watertight mesh."""
    verts = []
    faces = []
    n_rings = len(rings)
    pts_per_ring = len(rings[0])

    for ring in rings:
        for pt in ring:
            verts.append(pt)

    for i in range(n_rings - 1):
        r0 = i * pts_per_ring
        r1 = (i + 1) * pts_per_ring
        lim = pts_per_ring if closed_ring else pts_per_ring - 1
        for j in range(lim):
            jn = (j + 1) % pts_per_ring
            faces.append([r0 + j, r0 + jn, r1 + jn])
            faces.append([r0 + j, r1 + jn, r1 + j])

    if closed_ring:
        c0 = len(verts)
        verts.append(rings[0].mean(axis=0))
        for j in range(pts_per_ring):
            faces.append([c0, (j + 1) % pts_per_ring, j])
        c1 = len(verts)
        verts.append(rings[-1].mean(axis=0))
        lr = (n_rings - 1) * pts_per_ring
        for j in range(pts_per_ring):
            faces.append([c1, lr + j, lr + ((j + 1) % pts_per_ring)])

    return trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64), process=True)


def make_strand(curve: np.ndarray, rx: np.ndarray, ry: np.ndarray,
                profile_type: str = "crescent", radial_segs: int = 16) -> trimesh.Trimesh:
    """Extrude a 3D anime hair strand along a curve with curved cross-section."""
    pts = np.asarray(curve, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)
    ry = np.asarray(ry, dtype=np.float64)
    n = len(pts)

    tangents = np.zeros_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents[1:-1] = pts[2:] - pts[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents /= norms

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], up)) > 0.85:
        up = np.array([0.0, 1.0, 0.0])
    n0 = np.cross(tangents[0], up)
    n0 /= np.linalg.norm(n0)
    b0 = np.cross(tangents[0], n0)

    frames = [(n0, b0)]
    for i in range(1, n):
        t_prev, t_curr = tangents[i - 1], tangents[i]
        axis = np.cross(t_prev, t_curr)
        axis_len = np.linalg.norm(axis)
        n_prev, b_prev = frames[-1]
        if axis_len > 1e-6:
            axis /= axis_len
            ang = np.arccos(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
            def rot(v):
                return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * np.dot(axis, v) * (1.0 - np.cos(ang))
            frames.append((rot(n_prev), rot(b_prev)))
        else:
            frames.append((n_prev, b_prev))

    angles = np.linspace(0, 2 * np.pi, radial_segs, endpoint=False)
    verts, faces = [], []

    for i in range(n):
        p = pts[i]
        norm, binorm = frames[i]
        for a in angles:
            ca = np.cos(a)
            sa = np.sin(a)
            if profile_type == "crescent":
                px = ca
                py = sa + 0.35 * (ca**2 - 0.5)
            elif profile_type == "oval":
                px, py = ca, sa
            else:
                px, py = ca, sa
            offset = rx[i] * px * norm + ry[i] * py * binorm
            verts.append(p + offset)

    for i in range(n - 1):
        r0 = i * radial_segs
        r1 = (i + 1) * radial_segs
        for j in range(radial_segs):
            jn = (j + 1) % radial_segs
            faces.append([r0 + j, r0 + jn, r1 + jn])
            faces.append([r0 + j, r1 + jn, r1 + j])

    c_s = len(verts)
    verts.append(pts[0])
    for j in range(radial_segs):
        faces.append([c_s, (j + 1) % radial_segs, j])
    c_e = len(verts)
    verts.append(pts[-1])
    lr = (n - 1) * radial_segs
    for j in range(radial_segs):
        faces.append([c_e, lr + j, lr + ((j + 1) % radial_segs)])

    return trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64), process=True)


# --------------------------------------------------------------------------- #
# 1. Hair Base Cap (Full dome enclosing crown, sides, and nape)
# --------------------------------------------------------------------------- #
def build_skull_cap() -> trimesh.Trimesh:
    """Build a complete anime hair dome enclosing the crown, sides, and rear of the skull."""
    rings = []
    n_rings = 24
    pts_per_ring = 32

    for i in range(n_rings):
        t = i / (n_rings - 1)
        # Z from apex (77.6) down to nape of neck (57.5)
        z = 77.6 - 20.1 * t

        sin_prof = np.sin(np.pi * (0.02 + 0.98 * t)**0.6)
        rx = (8.5 * sin_prof + 0.2)
        if t > 0.65:
            rx *= (1.0 - 0.22 * (t - 0.65) / 0.35)

        y_center = 2.5 + 1.2 * (1.0 - t)
        ry_back = (8.8 * sin_prof + 0.3)
        if t > 0.72:
            ry_back *= (1.0 - 0.35 * (t - 0.72) / 0.28)

        ry_front = (6.6 * sin_prof + 0.1)

        ring = []
        for j in range(pts_per_ring):
            ang = 2 * np.pi * j / pts_per_ring
            ca = np.cos(ang)
            sa = np.sin(ang)

            x = rx * ca
            if sa >= 0:  # Back & sides
                y = y_center + ry_back * sa
            else:  # Front hairline
                front_blend = np.clip(1.0 - 1.1 * t, 0.25, 1.0)
                y = y_center + ry_front * sa * front_blend
            ring.append([x, y, z])
        rings.append(np.asarray(ring))

    return make_lofted_surface(rings, closed_ring=True)


# --------------------------------------------------------------------------- #
# 2. Kitsune / Fox Beast Ears
# --------------------------------------------------------------------------- #
def build_kitsune_ear(side: float = 1.0) -> trimesh.Trimesh:
    """Build large, stylized triangular kitsune ears with curved outer rim."""
    rings = []
    n_levels = 22
    pts_per_ring = 28

    for i in range(n_levels):
        t = i / (n_levels - 1)
        z = 74.8 + 13.5 * t  # Z up to 88.3
        # Tilt ear outward (~22 degrees) and slightly back
        x_center = side * (4.5 + 2.8 * t - 0.5 * t**2)
        y_center = 1.8 - 1.5 * t + 0.4 * t**2

        # Cross section dimensions: wide at base, tapering to sharp curved tip
        w = 4.4 * (1.0 - t**0.85) + 0.12
        d_back = 3.0 * (1.0 - t**0.8) + 0.08
        d_front = 2.0 * (1.0 - t) + 0.05

        ring = []
        for j in range(pts_per_ring):
            ang = 2 * np.pi * j / pts_per_ring
            ca = np.cos(ang)
            sa = np.sin(ang)

            if sa >= 0:  # Outer convex shell
                dx = side * ca * w * 1.05
                dy = sa * d_back
            else:  # Inner concave cavity
                dx = side * ca * w * 0.8
                dy = sa * d_front * 0.35  # Hollowed inward

            ring.append([x_center + dx, y_center + dy, z])
        rings.append(np.asarray(ring))

    return make_lofted_surface(rings, closed_ring=True)


def build_ear_fluff(side: float = 1.0) -> trimesh.Trimesh:
    """Build fluffy 3D fur tufts lining the inner base of the kitsune ear."""
    tufts = []
    configs = [
        ((side * 3.8, 0.8, 75.2), (side * 2.8, -0.6, 77.8), 0.95, 0.60),
        ((side * 4.8, 1.0, 76.5), (side * 3.8, -0.8, 80.0), 1.10, 0.65),
        ((side * 5.8, 1.2, 78.0), (side * 4.8, -0.5, 81.6), 0.90, 0.55),
        ((side * 4.2, 0.5, 76.0), (side * 3.2, -0.9, 78.8), 0.80, 0.50),
        ((side * 5.2, 0.6, 77.2), (side * 4.4, -0.7, 80.5), 0.75, 0.45),
    ]

    for root, tip, rx, ry in configs:
        t = np.linspace(0, 1, 12)
        curve = np.outer(1 - t, root) + np.outer(t, tip)
        curve[:, 1] -= 0.45 * np.sin(np.pi * t)  # puff outward-forward
        radii_x = rx * np.sin(np.pi * t**0.65) + 0.08
        radii_y = ry * np.sin(np.pi * t**0.65) + 0.05
        tufts.append(make_strand(curve, radii_x, radii_y, profile_type="oval", radial_segs=10))

    return trimesh.util.concatenate(tufts)


# --------------------------------------------------------------------------- #
# 3. Forehead Bangs (Volumetric Arched Fringe)
# --------------------------------------------------------------------------- #
def build_bangs() -> trimesh.Trimesh:
    """Build lush, arched forehead bangs hugging the brow with center notch."""
    bangs = []

    # Accurate forehead positions:
    # At Z=75: forehead Y is -4.4
    # At Z=70: forehead Y is -5.6
    # At Z=66: forehead Y is -6.2
    # At Z=64: forehead Y is -6.8
    # Bangs sit directly on the forehead (Y ~ -4.8 at top down to -7.1 at brow tips)
    strands = [
        # (center-left main bang)
        {"xs": -0.3, "xe": -1.4, "ys": -4.2, "ye": -6.9, "zs": 75.5, "ze": 65.2, "rx": 1.7, "ry": 0.70},
        # (center-right main bang)
        {"xs": 0.3, "xe": 1.4, "ys": -4.2, "ye": -6.9, "zs": 75.5, "ze": 65.2, "rx": 1.7, "ry": 0.70},
        # (center separator strand)
        {"xs": 0.0, "xe": 0.0, "ys": -4.5, "ye": -7.0, "zs": 75.0, "ze": 66.5, "rx": 0.9, "ry": 0.50},
        # (mid-left arched bang)
        {"xs": -1.5, "xe": -3.2, "ys": -4.0, "ye": -6.7, "zs": 75.2, "ze": 64.6, "rx": 1.6, "ry": 0.65},
        # (mid-right arched bang)
        {"xs": 1.5, "xe": 3.2, "ys": -4.0, "ye": -6.7, "zs": 75.2, "ze": 64.6, "rx": 1.6, "ry": 0.65},
        # (outer-left temple transition)
        {"xs": -3.0, "xe": -5.0, "ys": -3.5, "ye": -6.3, "zs": 74.8, "ze": 63.8, "rx": 1.4, "ry": 0.60},
        # (outer-right temple transition)
        {"xs": 3.0, "xe": 5.0, "ys": -3.5, "ye": -6.3, "zs": 74.8, "ze": 63.8, "rx": 1.4, "ry": 0.60},
    ]

    for s in strands:
        t = np.linspace(0, 1, 16)
        x = (1 - t) * s["xs"] + t * s["xe"]
        y_base = (1 - t) * s["ys"] + t * s["ye"]
        # Gentle puff over forehead curve
        y = y_base - 0.75 * np.sin(np.pi * t)
        z = (1 - t) * s["zs"] + t * s["ze"]

        # Curled-under tips at the bottom
        y[-3:] += np.array([0.2, 0.45, 0.7])
        z[-3:] += np.array([0.1, 0.25, 0.5])

        curve = np.column_stack([x, y, z])
        rx = s["rx"] * (0.6 + 0.8 * np.sin(np.pi * t**0.75))
        ry = s["ry"] * (0.5 + 0.7 * np.sin(np.pi * t**0.75))
        bangs.append(make_strand(curve, rx, ry, profile_type="crescent", radial_segs=12))

    return trimesh.util.concatenate(bangs)


# --------------------------------------------------------------------------- #
# 4. Face-Framing Curls (Cheek & Jaw Hugging Locks)
# --------------------------------------------------------------------------- #
def build_framing_locks() -> trimesh.Trimesh:
    """Build wide, lush face-framing locks with prominent inward C-curls around the jaw."""
    locks = []

    for side in [-1.0, 1.0]:
        t = np.linspace(0, 1, 20)
        # From temple down around jaw
        x = side * (5.8 + 0.8 * np.sin(np.pi * t) - 1.0 * t**2)
        y = -3.8 - 2.8 * t + 1.2 * np.sin(np.pi * t)
        z = 71.0 - 18.5 * t  # Z down to 52.5

        # C-curl inward towards chin/throat
        curl = np.clip((t - 0.55) / 0.45, 0, 1)
        x -= side * 2.2 * curl**1.7
        y -= 1.8 * curl**1.7
        z += 1.8 * curl**2

        curve = np.column_stack([x, y, z])
        rx = 1.8 * np.sin(np.pi * t**0.55) + 0.2
        ry = 0.9 * np.sin(np.pi * t**0.55) + 0.12
        locks.append(make_strand(curve, rx, ry, profile_type="crescent", radial_segs=14))

        # Secondary outer wing lock
        t2 = np.linspace(0, 1, 14)
        x2 = side * (6.6 + 1.2 * np.sin(np.pi * t2))
        y2 = -2.5 - 2.8 * t2
        z2 = 69.0 - 13.5 * t2
        curl2 = np.clip((t2 - 0.6) / 0.4, 0, 1)
        x2 -= side * 1.4 * curl2**2
        y2 -= 1.2 * curl2**2
        curve2 = np.column_stack([x2, y2, z2])
        rx2 = 1.3 * np.sin(np.pi * t2**0.6) + 0.15
        ry2 = 0.65 * np.sin(np.pi * t2**0.6) + 0.1
        locks.append(make_strand(curve2, rx2, ry2, profile_type="crescent", radial_segs=12))

    return trimesh.util.concatenate(locks)


# --------------------------------------------------------------------------- #
# 5. Twin Braids (Plump Overlapping Weave Over Shoulders)
# --------------------------------------------------------------------------- #
def build_twin_braids() -> trimesh.Trimesh:
    """Build detailed 3D twin braids draped over the shoulders with ties and tails."""
    braids = []

    for side in [-1.0, 1.0]:
        start_pt = np.array([side * 4.6, -2.0, 58.5])
        end_pt = np.array([side * 3.6, -4.6, 42.0])  # front torso
        length = np.linalg.norm(end_pt - start_pt)
        axis = (end_pt - start_pt) / length

        up = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(axis, up)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)

        num_turns = 4.0
        omega = num_turns * 2 * np.pi
        t = np.linspace(0, 1, 60)
        radius = 0.65
        strand_r = 0.38

        # 3 intertwined strands forming the classic braid
        for phase in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
            u = t * omega + phase
            dx = radius * np.cos(u)
            dy = 0.75 * radius * np.sin(2 * u)
            pts = np.outer(1 - t, start_pt) + np.outer(t, end_pt)
            pts += np.outer(dx, e1) + np.outer(dy, e2)
            r_profile = strand_r * (1.0 - 0.25 * t)
            braids.append(make_strand(pts, r_profile, r_profile, profile_type="oval", radial_segs=10))

        # Braid tie ring (elastic band)
        tie = trimesh.creation.cylinder(radius=0.9, height=0.7, sections=18)
        rot_mat = trimesh.geometry.align_vectors([0, 0, 1], axis)
        tie.apply_transform(rot_mat)
        tie.apply_translation(end_pt)
        braids.append(tie)

        # Flowing loose hair tail below the braid tie
        tail_t = np.linspace(0, 1, 20)
        tail_x = side * (3.6 + 0.6 * np.sin(np.pi * tail_t))
        tail_y = -4.6 - 1.4 * tail_t + 0.6 * np.sin(np.pi * tail_t)
        tail_z = 42.0 - 9.5 * tail_t  # down to Z~32.5
        tail_curve = np.column_stack([tail_x, tail_y, tail_z])
        tail_rx = 0.55 * (1.0 - tail_t**0.6) + 0.08
        tail_ry = 0.35 * (1.0 - tail_t**0.6) + 0.05
        braids.append(make_strand(tail_curve, tail_rx, tail_ry, profile_type="crescent", radial_segs=10))

    return trimesh.util.concatenate(braids)


# --------------------------------------------------------------------------- #
# 6. Cascading Back Hair Mane
# --------------------------------------------------------------------------- #
def build_back_hair() -> trimesh.Trimesh:
    """Build a voluminous cascading cape of hair down the back with layered waves."""
    layers = []

    # Array of layered locks cascading from crown (Z~74) down past mid-back (Z~27)
    n_locks = 15
    for k in range(n_locks):
        frac = k / (n_locks - 1)  # 0 to 1 across left-to-right
        x_base = -8.5 + 17.0 * frac
        flare = (frac - 0.5) * 8.5

        t = np.linspace(0, 1, 28)
        x = x_base + flare * t**1.35
        # Hug back curvature smoothly from Z=74 down to Z=27
        # At Z=74: y ~ 8.8
        # At Z=68: y ~ 10.4
        # At Z=60: y ~ 8.2
        # At Z=45: y ~ 4.5
        # At Z=27: y ~ 3.5
        y = 8.8 + 1.8 * np.sin(np.pi * t**0.5) - 6.5 * t + 0.8 * np.sin(np.pi * t)
        z = 74.5 - 47.0 * t  # Z down to 27.5

        # Wing tip flicks at the outer edges (matching reference rear view)
        if abs(frac - 0.5) > 0.22:
            wing_dir = np.sign(frac - 0.5)
            x += wing_dir * 2.6 * t**3.2
            y += 1.4 * t**3.2

        curve = np.column_stack([x, y, z])
        rx = (1.9 - 0.5 * abs(frac - 0.5)) * np.sin(np.pi * t**0.5) + 0.18
        ry = 0.95 * np.sin(np.pi * t**0.5) + 0.12
        layers.append(make_strand(curve, rx, ry, profile_type="crescent", radial_segs=12))

    return trimesh.util.concatenate(layers)


# --------------------------------------------------------------------------- #
# Complete Hairstyle Assembly
# --------------------------------------------------------------------------- #
def build_complete_hair() -> trimesh.Trimesh:
    """Assemble all hair parts into a single watertight combined mesh."""
    parts = []

    print("Building skull cap dome ...")
    parts.append(build_skull_cap())

    print("Building Kitsune beast ears ...")
    parts.append(build_kitsune_ear(side=1.0))
    parts.append(build_kitsune_ear(side=-1.0))
    parts.append(build_ear_fluff(side=1.0))
    parts.append(build_ear_fluff(side=-1.0))

    print("Building lush forehead bangs ...")
    parts.append(build_bangs())

    print("Building face-framing C-curls ...")
    parts.append(build_framing_locks())

    print("Building twin braids ...")
    parts.append(build_twin_braids())

    print("Building cascading back hair mane ...")
    parts.append(build_back_hair())

    combined = trimesh.util.concatenate(parts)
    print(f"Hair total: vertices={len(combined.vertices)}, faces={len(combined.faces)}")
    return combined


# --------------------------------------------------------------------------- #
# SFW Anatomical Base Refinement
# --------------------------------------------------------------------------- #
def refine_anatomy_sfw(body: trimesh.Trimesh) -> trimesh.Trimesh:
    """Smooth and refine chest / torso anatomy into an elegant SFW base mesh.
    
    Eliminates any artificial tip protrusions in the chest region,
    smooths the pectoral and breast curvature into natural, graceful curves,
    and refines the clavicles and neckline for a clean, aesthetic sculpt.
    """
    verts = body.vertices.copy()
    faces = body.faces.copy()

    # Smooth breast curvature into natural SFW feminine forms
    for side in [-1.0, 1.0]:
        center = np.array([side * 3.6, -7.2, 48.8])
        dists = np.linalg.norm(verts - center, axis=1)
        tip_mask = dists < 2.5
        if np.any(tip_mask):
            for idx in np.where(tip_mask)[0]:
                d = dists[idx]
                w = np.cos(0.5 * np.pi * (d / 2.5)) ** 2
                if verts[idx, 1] < -7.3:
                    verts[idx, 1] = verts[idx, 1] * (1 - w) + (-7.3) * w

    # Clavicle / collarbone aesthetic refinement
    clavicle_mask = (verts[:, 2] >= 53.5) & (verts[:, 2] <= 55.5) & (verts[:, 1] <= -2.5) & (verts[:, 1] >= -5.0) & (np.abs(verts[:, 0]) >= 1.5) & (np.abs(verts[:, 0]) <= 7.5)
    if np.any(clavicle_mask):
        for idx in np.where(clavicle_mask)[0]:
            x_frac = (abs(verts[idx, 0]) - 1.5) / 6.0
            curve_profile = np.sin(np.pi * x_frac)
            verts[idx, 1] -= 0.15 * curve_profile

    refined = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return refined


if __name__ == "__main__":
    hair = build_complete_hair()
    hair.export("Chibi_hair.stl")
    print("Saved Chibi_hair.stl")

    body = trimesh.load("Chibi_full.stl")
    body_sfw = refine_anatomy_sfw(body)
    body_sfw.export("Chibi_body_sfw.stl")
    print("Saved Chibi_body_sfw.stl")

    combined = trimesh.util.concatenate([body_sfw, hair])
    combined.export("Chibi_s1_fused.stl")
    print("Saved Chibi_s1_fused.stl with", len(combined.faces), "triangles")
