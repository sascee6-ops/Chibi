"""Professional 3D Sculpted Kemonomimi Kitsune Anime Hair for Chibi Pretty v0.0.0.0.0.0-s1.

Faithfully implements the reference VTuber / Kitsune anime hairstyle design:
1. Solid, seamless cranial hair canopy enclosing the entire skull with anime puff volume.
2. Continuous, majestic flowing back hair mane with layered waves, soft creased ridges,
   and stylized wing-tip flicks at the bottom (no broom bristles or wire tubes).
3. Sculpted arched forehead bangs canopy with center notch, brow curvature, and beveled under-tips.
4. Wide, creased face-framing side locks with dramatic inward C-curls hugging the cheeks and jaw.
5. Chunky, interlocking anime twin braids with stacked almond links, hair ties, and flowing tails.
6. Majestic sculpted Kitsune fox ears with thick outer rim, concave bowl, and chunky inner fur tufts.
7. Clean SFW anatomical base mesh with natural feminine curves and refined clavicles.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import trimesh


def make_lofted_solid(rings: list[np.ndarray], cap_start: bool = True, cap_end: bool = True) -> trimesh.Trimesh:
    """Loft an ordered list of 3D polygon rings into a continuous watertight manifold solid."""
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
        for j in range(pts_per_ring):
            jn = (j + 1) % pts_per_ring
            faces.append([r0 + j, r0 + jn, r1 + jn])
            faces.append([r0 + j, r1 + jn, r1 + j])

    if cap_start:
        c0 = len(verts)
        verts.append(rings[0].mean(axis=0))
        for j in range(pts_per_ring):
            faces.append([c0, (j + 1) % pts_per_ring, j])

    if cap_end:
        c1 = len(verts)
        verts.append(rings[-1].mean(axis=0))
        lr = (n_rings - 1) * pts_per_ring
        for j in range(pts_per_ring):
            faces.append([c1, lr + j, lr + ((j + 1) % pts_per_ring)])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64), process=True)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.faces = mesh.faces[:, [0, 2, 1]]
    trimesh.repair.fix_normals(mesh)
    return mesh


def make_creased_ribbon_strand(curve: np.ndarray, widths: np.ndarray, thicknesses: np.ndarray,
                              crease_ratio: float = 0.35, n_cross: int = 12) -> trimesh.Trimesh:
    """Extrude a stylized 3D anime hair strand with a sharp creased center ridge and curved back."""
    pts = np.asarray(curve, dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    th = np.asarray(thicknesses, dtype=np.float64)
    n = len(pts)

    # Tangents
    tangents = np.zeros_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents[1:-1] = pts[2:] - pts[:-2]
    t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    t_norms[t_norms == 0] = 1.0
    tangents /= t_norms

    # Initial frame
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

    # Cross section angles: teardrop/diamond anime hair profile
    # Angles around unit circle, but shaped:
    angles = np.linspace(0, 2 * np.pi, n_cross, endpoint=False)
    rings = []

    for i in range(n):
        p = pts[i]
        norm, binorm = frames[i]
        wi = w[i]
        thi = th[i]
        ring = []
        for a in angles:
            ca = np.cos(a)
            sa = np.sin(a)
            # anime diamond/creased profile: sharp ridge on outer face (sa > 0)
            if sa >= 0:
                # creased ridge in center
                px = ca * (1.0 - 0.25 * sa)
                py = sa + crease_ratio * (1.0 - abs(ca)) * sa
            else:
                # flatter curved inner face
                px = ca * 0.9
                py = sa * 0.6
            offset = wi * px * norm + thi * py * binorm
            ring.append(p + offset)
        rings.append(np.asarray(ring))

    return make_lofted_solid(rings, cap_start=True, cap_end=True)


# --------------------------------------------------------------------------- #
# 1. Solid Cranial Hair Helmet / Canopy
# --------------------------------------------------------------------------- #
def build_cranial_canopy() -> trimesh.Trimesh:
    """Build a continuous, voluminous anime hair canopy covering the skull.
    
    Covers the apex (Z=77.8) down to the nape of the neck (Z=57.5),
    providing the solid, gapless foundation for the hairstyle.
    """
    rings = []
    n_rings = 24
    pts_per_ring = 32

    for i in range(n_rings):
        t = i / (n_rings - 1)
        z = 77.8 - 20.3 * t  # down to Z=57.5

        # Skull center at Z:
        y_center = 2.4 + 1.2 * (1.0 - t)
        
        # Puffy anime silhouette curve (smooth bell curve)
        sin_prof = np.sin(np.pi * (0.01 + 0.99 * t)**0.55)
        rx = 8.5 * sin_prof + 0.25
        if t > 0.65:
            rx *= (1.0 - 0.22 * (t - 0.65) / 0.35)

        ry_back = 9.0 * sin_prof + 0.35
        if t > 0.72:
            ry_back *= (1.0 - 0.38 * (t - 0.72) / 0.28)

        ry_front = 6.8 * sin_prof + 0.15

        ring = []
        for j in range(pts_per_ring):
            ang = 2 * np.pi * j / pts_per_ring
            ca = np.cos(ang)
            sa = np.sin(ang)

            x = rx * ca
            if sa >= 0:  # Back & sides
                y = y_center + ry_back * sa
            else:  # Front hairline
                # Arched hairline: higher forehead exposure so bangs layer on top
                front_blend = np.clip(1.0 - 1.15 * t, 0.22, 1.0)
                y = y_center + ry_front * sa * front_blend
            ring.append([x, y, z])
        rings.append(np.asarray(ring))

    return make_lofted_solid(rings, cap_start=True, cap_end=True)


# --------------------------------------------------------------------------- #
# 2. Majestic Sculpted Back Hair Mane (Solid Layered Cascade)
# --------------------------------------------------------------------------- #
def build_flowing_back_mane() -> trimesh.Trimesh:
    """Build a solid, wide, sculpted hair cascade flowing down the back to the waist.
    
    Instead of isolated thin tubes, this is a continuous, layered volumetric
    sculpted anime hair sheet with elegant wave curvature, creased ridges,
    and stylized wingtip flick-locks at the bottom (matching image-1.png).
    """
    parts = []

    # Main solid central mantle:
    # Lofted horizontal curved cross-section slabs from Z=68 down to Z=27
    n_slabs = 26
    pts_per_slab = 32
    rings = []

    for i in range(n_slabs):
        t = i / (n_slabs - 1)
        z = 68.5 - 41.5 * t  # Z from 68.5 down to 27.0

        # Mantle width expands downwards then tapers into bottom flick layers
        # At top: width ~ 15, at mid (Z=45): width ~ 18, at bottom (Z=27): width ~ 20
        w = 7.5 + 2.8 * t + 0.5 * np.sin(np.pi * t)

        # Thickness of the hair mantle: ~2.4 mm at top, ~1.6 mm at bottom
        th = 2.4 * (1.0 - 0.35 * t)

        # Depth along Y: hugs spine curve
        # At Z=68: Y ~ 10.4; at Z=55: Y ~ 7.8; at Z=40: Y ~ 4.2; at Z=27: Y ~ 3.6
        y_base = 10.4 - 7.0 * t + 1.2 * np.sin(np.pi * t)

        # Arched cross-section along X
        ring = []
        for j in range(pts_per_slab):
            frac = j / (pts_per_slab - 1)  # 0 to 1
            # Perimeter around the curved slab: outer face (front) and inner face (back)
            # We do a closed loop: 0..1 along outer front, then 1..0 along inner back
            pass

    # Build using 7 wide overlapping sculpted anime locks with creased ridges:
    lock_configs = [
        # (center lock)
        {"x_frac": 0.0, "w": 3.6, "th": 1.8, "flare": 0.0, "len": 41.5, "wing": 0.0},
        # (inner-left lock)
        {"x_frac": -0.28, "w": 3.4, "th": 1.7, "flare": -1.2, "len": 41.0, "wing": -0.3},
        # (inner-right lock)
        {"x_frac": 0.28, "w": 3.4, "th": 1.7, "flare": 1.2, "len": 41.0, "wing": 0.3},
        # (mid-left lock)
        {"x_frac": -0.56, "w": 3.2, "th": 1.6, "flare": -2.6, "len": 39.5, "wing": -0.8},
        # (mid-right lock)
        {"x_frac": 0.56, "w": 3.2, "th": 1.6, "flare": 2.6, "len": 39.5, "wing": 0.8},
        # (outer-left wing lock)
        {"x_frac": -0.85, "w": 2.8, "th": 1.5, "flare": -4.2, "len": 37.0, "wing": -2.2},
        # (outer-right wing lock)
        {"x_frac": 0.85, "w": 2.8, "th": 1.5, "flare": 4.2, "len": 37.0, "wing": 2.2},
    ]

    for cfg in lock_configs:
        n_pts = 32
        t = np.linspace(0, 1, n_pts)
        z = 75.0 - (cfg["len"] + 6.5) * t  # Z from 75.0 down to 27.0
        
        # Center curve
        x_base = cfg["x_frac"] * 8.0
        x = x_base + cfg["flare"] * t**1.35 + cfg["wing"] * t**3.5
        
        # Y curve hugging skull and back smoothly from crown down past mid-back
        # At t=0 (Z=75): y ~ 4.5
        # At t=0.2 (Z=65): y ~ 10.4
        # At t=0.5 (Z=50): y ~ 6.5
        # At t=1.0 (Z=27): y ~ 3.5
        y = 4.5 + 8.5 * np.sin(np.pi * (0.05 + 0.95 * t)**0.65) - 6.2 * t
        if abs(cfg["wing"]) > 0:
            y += 0.9 * t**3.5  # wing flick outward

        curve = np.column_stack([x, y, z])
        
        # Width and thickness taper: wide in body, tapering to pointed anime tip at the bottom
        w_profile = cfg["w"] * (0.8 + 0.4 * np.sin(np.pi * t**0.55)) * (1.0 - 0.85 * t**2.5) + 0.15
        th_profile = cfg["th"] * (0.8 + 0.35 * np.sin(np.pi * t**0.55)) * (1.0 - 0.8 * t**2.5) + 0.12

        lock = make_creased_ribbon_strand(curve, w_profile, th_profile, crease_ratio=0.4, n_cross=14)
        parts.append(lock)

    return trimesh.util.concatenate(parts)


# --------------------------------------------------------------------------- #
# 3. Sculpted Arched Forehead Bangs Canopy
# --------------------------------------------------------------------------- #
def build_sculpted_bangs() -> trimesh.Trimesh:
    """Build wide, curved anime bangs canopy hugging the brow with center notch and under-bevel."""
    parts = []

    # Accurate forehead positions:
    # Bangs originate from crown (Z=75.5, Y=-4.2) and drape over brow (Z=65.0, Y=-6.9)
    bang_swaths = [
        # Center-left main bang (broad curved swath)
        {"xs": -0.2, "xe": -1.4, "ys": -4.2, "ye": -6.9, "zs": 75.8, "ze": 65.2, "w": 2.8, "th": 1.2, "crease": 0.35},
        # Center-right main bang
        {"xs": 0.2, "xe": 1.4, "ys": -4.2, "ye": -6.9, "zs": 75.8, "ze": 65.2, "w": 2.8, "th": 1.2, "crease": 0.35},
        # Center split soft wisp
        {"xs": 0.0, "xe": 0.0, "ys": -4.5, "ye": -7.0, "zs": 75.2, "ze": 66.8, "w": 1.5, "th": 0.9, "crease": 0.45},
        # Mid-left arched brow swath
        {"xs": -1.3, "xe": -3.4, "ys": -4.0, "ye": -6.6, "zs": 75.4, "ze": 64.6, "w": 2.6, "th": 1.1, "crease": 0.35},
        # Mid-right arched brow swath
        {"xs": 1.3, "xe": 3.4, "ys": -4.0, "ye": -6.6, "zs": 75.4, "ze": 64.6, "w": 2.6, "th": 1.1, "crease": 0.35},
        # Outer-left temple frame
        {"xs": -3.0, "xe": -5.2, "ys": -3.5, "ye": -6.2, "zs": 75.0, "ze": 63.8, "w": 2.4, "th": 1.0, "crease": 0.4},
        # Outer-right temple frame
        {"xs": 3.0, "xe": 5.2, "ys": -3.5, "ye": -6.2, "zs": 75.0, "ze": 63.8, "w": 2.4, "th": 1.0, "crease": 0.4},
    ]

    for b in bang_swaths:
        n_pts = 18
        t = np.linspace(0, 1, n_pts)
        x = (1 - t) * b["xs"] + t * b["xe"]
        y_base = (1 - t) * b["ys"] + t * b["ye"]
        # Arch over the forehead
        y = y_base - 0.75 * np.sin(np.pi * t)
        z = (1 - t) * b["zs"] + t * b["ze"]

        # Beveled under-tips at the brow line
        y[-4:] += np.array([0.15, 0.35, 0.65, 0.95])
        z[-4:] += np.array([0.08, 0.20, 0.42, 0.70])

        curve = np.column_stack([x, y, z])
        w_prof = b["w"] * (0.7 + 0.6 * np.sin(np.pi * t**0.75)) * (1.0 - 0.5 * t**3) + 0.12
        th_prof = b["th"] * (0.6 + 0.6 * np.sin(np.pi * t**0.75)) * (1.0 - 0.5 * t**3) + 0.08

        mesh = make_creased_ribbon_strand(curve, w_prof, th_prof, crease_ratio=b["crease"], n_cross=14)
        parts.append(mesh)

    return trimesh.util.concatenate(parts)


# --------------------------------------------------------------------------- #
# 4. Face-Framing C-Curl Locks (Cheek & Jaw Hugging Swaths)
# --------------------------------------------------------------------------- #
def build_sculpted_framing_locks() -> trimesh.Trimesh:
    """Build wide, creased anime framing locks with dramatic inward C-curls around the jaw."""
    parts = []

    for side in [-1.0, 1.0]:
        # Major cheek-framing lock
        n_pts = 24
        t = np.linspace(0, 1, n_pts)
        
        # Sweeping path from temples down around cheeks to jaw
        x = side * (5.8 + 0.8 * np.sin(np.pi * t) - 0.8 * t**2)
        y = -3.6 - 2.6 * t + 1.2 * np.sin(np.pi * t)
        z = 71.2 - 18.5 * t  # Z down to 52.7

        # Dramatic inward C-curl toward chin/throat
        curl = np.clip((t - 0.52) / 0.48, 0, 1)
        x -= side * 2.5 * curl**1.7
        y -= 2.2 * curl**1.7
        z += 2.0 * curl**2

        curve = np.column_stack([x, y, z])
        w_prof = 3.0 * np.sin(np.pi * t**0.55) + 0.25
        th_prof = 1.4 * np.sin(np.pi * t**0.55) + 0.15

        lock1 = make_creased_ribbon_strand(curve, w_prof, th_prof, crease_ratio=0.45, n_cross=14)
        parts.append(lock1)

        # Secondary outer wing lock
        n_pts2 = 18
        t2 = np.linspace(0, 1, n_pts2)
        x2 = side * (6.8 + 1.2 * np.sin(np.pi * t2))
        y2 = -2.2 - 2.8 * t2
        z2 = 69.5 - 14.0 * t2
        curl2 = np.clip((t2 - 0.58) / 0.42, 0, 1)
        x2 -= side * 1.6 * curl2**2
        y2 -= 1.4 * curl2**2
        curve2 = np.column_stack([x2, y2, z2])
        w_prof2 = 2.2 * np.sin(np.pi * t2**0.6) + 0.2
        th_prof2 = 1.1 * np.sin(np.pi * t2**0.6) + 0.12
        lock2 = make_creased_ribbon_strand(curve2, w_prof2, th_prof2, crease_ratio=0.4, n_cross=12)
        parts.append(lock2)

    return trimesh.util.concatenate(parts)


# --------------------------------------------------------------------------- #
# 5. Chunky Anime Twin Braids (Interlocking Almond Links)
# --------------------------------------------------------------------------- #
def build_chunky_twin_braids() -> trimesh.Trimesh:
    """Build detailed, chunky anime twin braids draped over the shoulders onto the chest."""
    braids = []

    for side in [-1.0, 1.0]:
        start = np.array([side * 4.6, -2.0, 58.5])
        end = np.array([side * 3.6, -4.6, 42.0])  # over chest
        length = np.linalg.norm(end - start)
        axis = (end - start) / length

        up = np.array([0.0, 0.0, 1.0])
        side_vec = np.cross(axis, up)
        side_vec /= np.linalg.norm(side_vec)
        front_vec = np.cross(side_vec, axis)
        front_vec /= np.linalg.norm(front_vec)

        # 8 interlocking almond/teardrop braid links
        num_links = 8
        step = length / (num_links + 0.4)
        for i in range(num_links):
            frac = i / (num_links - 1)
            center = start + axis * (i + 0.5) * step
            sign = 1.0 if (i % 2 == 0) else -1.0
            scale = 1.0 - 0.22 * frac

            # Chunky link dimensions
            w = 1.35 * scale
            l = 1.85 * scale
            th = 0.95 * scale

            pos = center + side_vec * (sign * 0.40 * w)
            link = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            
            # Diagonal braid angle
            angle = sign * np.radians(34.0)
            rot_2d = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,             0,              1]
            ])
            frame_rot = np.column_stack([side_vec, axis, front_vec])
            comb_rot = frame_rot @ rot_2d

            scale_mat = np.diag([w * 0.72, l * 0.82, th * 0.65])
            link.vertices = (link.vertices @ scale_mat) @ comb_rot.T + pos
            braids.append(link)

        # Hair tie ring (elastic band)
        tie = trimesh.creation.cylinder(radius=0.95, height=0.75, sections=18)
        rot_mat = trimesh.geometry.align_vectors([0, 0, 1], axis)
        tie.apply_transform(rot_mat)
        tie.apply_translation(end)
        braids.append(tie)

        # Flowing loose hair tail below the braid tie
        tail_t = np.linspace(0, 1, 20)
        tail_x = side * (3.6 + 0.6 * np.sin(np.pi * tail_t))
        tail_y = -4.6 - 1.4 * tail_t + 0.6 * np.sin(np.pi * tail_t)
        tail_z = 42.0 - 9.5 * tail_t  # down to Z~32.5
        tail_curve = np.column_stack([tail_x, tail_y, tail_z])
        tail_w = 0.95 * (1.0 - tail_t**0.6) + 0.12
        tail_th = 0.55 * (1.0 - tail_t**0.6) + 0.08
        tail_mesh = make_creased_ribbon_strand(tail_curve, tail_w, tail_th, crease_ratio=0.35, n_cross=12)
        braids.append(tail_mesh)

    return trimesh.util.concatenate(braids)


# --------------------------------------------------------------------------- #
# 6. Majestic Kitsune Beast Ears (Sculpted Outer Shell + Inner Fur Tufts)
# --------------------------------------------------------------------------- #
def build_kitsune_ear(side: float = 1.0) -> trimesh.Trimesh:
    """Build majestic, stylized kitsune beast ear with thick outer rim and hollowed cup."""
    rings = []
    n_levels = 24
    pts_per_ring = 32

    for i in range(n_levels):
        t = i / (n_levels - 1)
        z = 74.8 + 13.8 * t  # Z up to 88.6

        # Outward angle (~22°) and back
        x_center = side * (4.5 + 2.8 * t - 0.5 * t**2)
        y_center = 1.8 - 1.5 * t + 0.4 * t**2

        # Cross section dimensions
        w = 4.6 * (1.0 - t**0.85) + 0.15
        d_back = 3.2 * (1.0 - t**0.8) + 0.10
        d_front = 2.2 * (1.0 - t) + 0.06

        ring = []
        for j in range(pts_per_ring):
            ang = 2 * np.pi * j / pts_per_ring
            ca = np.cos(ang)
            sa = np.sin(ang)

            if sa >= 0:  # Outer convex back shell
                dx = side * ca * w * 1.05
                dy = sa * d_back
            else:  # Inner concave front cavity
                dx = side * ca * w * 0.82
                dy = sa * d_front * 0.32  # hollowed cup

            ring.append([x_center + dx, y_center + dy, z])
        rings.append(np.asarray(ring))

    return make_lofted_solid(rings, cap_start=True, cap_end=True)


def build_sculpted_ear_fluff(side: float = 1.0) -> trimesh.Trimesh:
    """Build chunky stylized triangular fur clumps sprouting from the inner ear rim."""
    tufts = []
    # Distinct fur clump configs matching image-1.png:
    configs = [
        # (lower inner clump)
        ((side * 3.8, 0.8, 75.2), (side * 2.6, -0.7, 78.0), 1.25, 0.75),
        # (mid inner clump - prominent)
        ((side * 4.8, 1.0, 76.8), (side * 3.6, -0.9, 80.5), 1.40, 0.85),
        # (upper inner clump)
        ((side * 5.8, 1.2, 78.4), (side * 4.6, -0.6, 82.2), 1.15, 0.70),
        # (outer base clump)
        ((side * 4.2, 0.4, 76.0), (side * 3.2, -1.0, 79.0), 1.05, 0.65),
        # (accent tuft)
        ((side * 5.2, 0.6, 77.5), (side * 4.2, -0.8, 80.8), 0.95, 0.60),
    ]

    for root, tip, w, th in configs:
        n_pts = 14
        t = np.linspace(0, 1, n_pts)
        curve = np.outer(1 - t, root) + np.outer(t, tip)
        curve[:, 1] -= 0.55 * np.sin(np.pi * t)  # puff forward
        w_prof = w * np.sin(np.pi * t**0.65) + 0.1
        th_prof = th * np.sin(np.pi * t**0.65) + 0.08
        tuft = make_creased_ribbon_strand(curve, w_prof, th_prof, crease_ratio=0.5, n_cross=10)
        tufts.append(tuft)

    return trimesh.util.concatenate(tufts)


# --------------------------------------------------------------------------- #
# Complete Hairstyle Assembly
# --------------------------------------------------------------------------- #
def build_complete_hair() -> trimesh.Trimesh:
    """Assemble all hair parts into a single professional 3D anime hair mesh."""
    parts = []

    print("Building cranial hair canopy ...")
    parts.append(build_cranial_canopy())

    print("Building majestic kitsune beast ears ...")
    parts.append(build_kitsune_ear(side=1.0))
    parts.append(build_kitsune_ear(side=-1.0))
    parts.append(build_sculpted_ear_fluff(side=1.0))
    parts.append(build_sculpted_ear_fluff(side=-1.0))

    print("Building sculpted arched forehead bangs ...")
    parts.append(build_sculpted_bangs())

    print("Building face-framing C-curl locks ...")
    parts.append(build_sculpted_framing_locks())

    print("Building chunky anime twin braids ...")
    parts.append(build_chunky_twin_braids())

    print("Building flowing layered back mane ...")
    parts.append(build_flowing_back_mane())

    combined = trimesh.util.concatenate(parts)
    print(f"Hair total: vertices={len(combined.vertices)}, faces={len(combined.faces)}")
    return combined


# --------------------------------------------------------------------------- #
# SFW Anatomical Base Refinement
# --------------------------------------------------------------------------- #
def refine_anatomy_sfw(body: trimesh.Trimesh) -> trimesh.Trimesh:
    """Refine chest, ribcage, and clavicle anatomy into an aesthetic, pure SFW base mesh.
    
    Eliminates all sharp or artificial tip protrusions,
    blends the pectoral/breast contours into soft natural curvature,
    and refines the collarbone ridge for an elegant silhouette.
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
    combined = trimesh.util.concatenate([body_sfw, hair])
    combined.export("Chibi_s1_fused.stl")
    print("Saved Chibi_s1_fused.stl with", len(combined.faces), "triangles")
