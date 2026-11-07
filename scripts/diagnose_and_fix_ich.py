"""Comprehensive diagnosis and fix for all black spots in ICHxTenebra.

Applies the Dewy Satin Porcelain skin painting, procedural shader, and textures
directly to the character mesh in 3D object space, eliminating all UV seams,
black margin padding, and tangent normal inversions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="ICHxTenebra.v0.0.0.0.0.0-s0.blend")
    p.add_argument("--out", default="ICHxTenebra_with_skin.blend")
    p.add_argument("--report", default="preview/ich_diagnostics.txt")
    p.add_argument("--render-previews", action="store_true")
    p.add_argument("--out-dir", default=".")
    return p.parse_args(argv)


def set_input(node, name: str, value) -> None:
    if name in node.inputs:
        node.inputs[name].default_value = value


def hex_srgb_to_linear(hex_color: str) -> np.ndarray:
    clean = hex_color.strip().lstrip("#")
    r, g, b = (int(clean[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    rgb = np.array([r, g, b], dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.append(linear, 1.0)


# 2026 Cinematic Dewy Satin Porcelain Palette
SKIN_HIGH = "#FFE8D6"    # delicate glowing ivory-peach for upper body / face
SKIN_MAIN = "#FFD9C2"    # healthy, warm peachy porcelain midtone
SKIN_LOW = "#FFC8AA"     # warm golden undertone for legs and feet
SKIN_GLOW = "#FFF3E6"    # dewy glass-skin highlight (forehead, nose bridge, cheek apex)
SKIN_COOL = "#EFC2B2"    # gentle cool shadow undertone

# Blush & Facial Features
BLUSH_CORE = "#FF7C98"   # fresh, lively rose core of the blush
BLUSH_HALO = "#FFAA94"   # soft warm peachy-coral glow radiating outwards
EYELIDS = "#FFAEC0"      # gentle romantic warmth on the lids
NOSE_TIP = "#FF8282"     # adorable soft flush on the nose tip
LIPS = "#F25D82"         # juicy, glazed berry-pink lips
LIP_CORE = "#D93B66"     # subtle depth at the mouth center
BEAUTY_MARK = "#7A4A32"  # delicate soft-brown beauty mark
FRECKLE = "#C48E66"      # warm translucent golden-cinnamon freckles
WARM_BODY = "#FFAF99"    # gentle warmth for collarbones, elbows, knuckles, knees
FINGERTIPS = "#FFB8CA"   # soft blushing fingertips and toes


def _blob(verts: np.ndarray, normals: np.ndarray, center, radii, strength: float, need_front=True) -> np.ndarray:
    centre = np.asarray(center, dtype=np.float64)
    radius = np.asarray(radii, dtype=np.float64)
    d2 = np.sum(((verts - centre) / radius) ** 2, axis=1)
    weights = strength * np.exp(-2.2 * d2)
    if need_front:
        frontness = np.clip((-normals[:, 1] - 0.35) / 0.40, 0.0, 1.0)
        weights *= frontness
    return np.clip(weights, 0.0, 1.0)


def paint_skin_colors(obj: bpy.types.Object) -> dict:
    mesh = obj.data
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)

    normals = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)

    count = len(verts)
    colors = np.empty((count, 4), dtype=np.float64)

    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    t = np.clip((verts[:, 2] - z_min) / max(z_max - z_min, 1e-9), 0.0, 1.0)
    low = hex_srgb_to_linear(SKIN_LOW)
    main = hex_srgb_to_linear(SKIN_MAIN)
    high = hex_srgb_to_linear(SKIN_HIGH)
    for channel in range(4):
        colors[:, channel] = np.where(
            t < 0.5,
            low[channel] + (main[channel] - low[channel]) * (t / 0.5),
            main[channel] + (high[channel] - main[channel]) * ((t - 0.5) / 0.5),
        )

    def apply_blob(center, radii, hex_color, strength, need_front=True):
        weights = _blob(verts, normals, center, radii, strength, need_front)
        color = hex_srgb_to_linear(hex_color)
        colors[:] = colors * (1.0 - weights[:, None]) + color[None, :] * weights[:, None]
        return int((weights > 0.05).sum())

    rough = np.full((count, 4), 0.47, dtype=np.float64)
    rough[:, 3] = 1.0

    def apply_rough_blob(center, radii, target, strength, need_front=True):
        weights = _blob(verts, normals, center, radii, strength, need_front)
        rough[:, :3] = (
            rough[:, :3] * (1.0 - weights[:, None])
            + float(target) * weights[:, None]
        )
        return int((weights > 0.05).sum())

    painted = {}
    glow = hex_srgb_to_linear(SKIN_GLOW)
    top_weights = 0.12 * np.clip((normals[:, 2] - 0.10) / 0.70, 0.0, 1.0) ** 1.4
    colors[:] = colors * (1.0 - top_weights[:, None]) + glow[None, :] * top_weights[:, None]
    painted["top_light"] = int((top_weights > 0.02).sum())

    cool = hex_srgb_to_linear(SKIN_COOL)
    shade_weights = 0.07 * np.clip((-normals[:, 2] - 0.20) / 0.55, 0.0, 1.0) ** 1.4
    colors[:] = colors * (1.0 - shade_weights[:, None]) + cool[None, :] * shade_weights[:, None]
    painted["cool_shade"] = int((shade_weights > 0.02).sum())

    # Layered living blush
    painted["face_glow"] = apply_blob((0.0, -6.35, 64.75), (4.2, 2.6, 2.8), SKIN_GLOW, 0.24)
    painted["forehead_glow"] = apply_blob((0.0, -5.7, 68.1), (3.5, 2.8, 1.8), SKIN_GLOW, 0.18)
    painted["cheek_halo"] = apply_blob((-3.15, -6.1, 63.8), (2.35, 2.8, 1.6), BLUSH_HALO, 0.38)
    painted["cheek_halo"] += apply_blob((3.15, -6.1, 63.8), (2.35, 2.8, 1.6), BLUSH_HALO, 0.38)
    painted["cheek_left"] = apply_blob((-3.15, -6.1, 63.8), (1.55, 2.2, 1.05), BLUSH_CORE, 0.74)
    painted["cheek_right"] = apply_blob((3.15, -6.1, 63.8), (1.55, 2.2, 1.05), BLUSH_CORE, 0.74)
    painted["eyelids"] = apply_blob((-2.4, -5.9, 66.0), (1.1, 1.8, 0.55), EYELIDS, 0.28)
    painted["eyelids"] += apply_blob((2.4, -5.9, 66.0), (1.1, 1.8, 0.55), EYELIDS, 0.28)
    painted["nose"] = apply_blob((0.05, -7.25, 63.95), (0.55, 0.9, 0.50), NOSE_TIP, 0.58)
    painted["lips"] = apply_blob((0.0, -6.75, 62.75), (0.60, 1.0, 0.38), LIPS, 0.65)
    painted["lip_core"] = apply_blob((0.0, -6.80, 62.75), (0.35, 0.8, 0.22), LIP_CORE, 0.35)
    painted["beauty_mark"] = apply_blob((1.75, -6.85, 63.08), (0.13, 0.26, 0.13), BEAUTY_MARK, 0.55)

    # Dewy Glass Skin roughness zones
    painted["rough_face"] = apply_rough_blob((0.0, -6.35, 64.9), (4.6, 2.7, 3.0), 0.40, 0.50)
    painted["rough_cheeks"] = apply_rough_blob((-3.15, -6.1, 63.8), (1.9, 2.4, 1.3), 0.38, 0.40)
    painted["rough_cheeks"] += apply_rough_blob((3.15, -6.1, 63.8), (1.9, 2.4, 1.3), 0.38, 0.40)
    painted["rough_eyelids"] = apply_rough_blob((-2.4, -5.9, 66.0), (1.15, 1.8, 0.6), 0.33, 0.45)
    painted["rough_eyelids"] += apply_rough_blob((2.4, -5.9, 66.0), (1.15, 1.8, 0.6), 0.33, 0.45)
    painted["rough_nose"] = apply_rough_blob((0.05, -7.25, 63.95), (0.7, 1.0, 0.6), 0.32, 0.80)
    painted["rough_lips"] = apply_rough_blob((0.0, -6.75, 62.75), (0.65, 1.05, 0.42), 0.22, 0.90)
    painted["rough_clavicle"] = apply_rough_blob((0.0, -4.5, 52.0), (9.0, 4.0, 2.5), 0.36, 0.35)

    # Warm accents
    painted["hands"] = apply_blob((-22.0, 0.0, 36.0), (2.5, 3.5, 4.0), WARM_BODY, 0.20, False)
    painted["hands"] += apply_blob((22.0, 0.0, 36.0), (2.5, 3.5, 4.0), WARM_BODY, 0.20, False)
    painted["knees"] = apply_blob((-7.5, -4.3, 17.0), (1.8, 1.5, 2.0), WARM_BODY, 0.16)
    painted["knees"] += apply_blob((7.5, -4.3, 17.0), (1.8, 1.5, 2.0), WARM_BODY, 0.16)
    feet_mask = verts[:, 2] < (z_min + 2.5)
    warm = hex_srgb_to_linear(WARM_BODY)
    colors[feet_mask] = colors[feet_mask] * 0.88 + warm[None, :] * 0.12
    rough[feet_mask, :3] = rough[feet_mask, :3] * 0.72 + 0.58 * 0.28
    painted["feet"] = int(feet_mask.sum())
    painted["rough_hands"] = apply_rough_blob((-22.0, 0.0, 36.0), (2.6, 3.6, 4.2), 0.39, 0.28, False)
    painted["rough_hands"] += apply_rough_blob((22.0, 0.0, 36.0), (2.6, 3.6, 4.2), 0.39, 0.28, False)

    # Butterfly freckles
    rng = np.random.default_rng(42)
    n_cheeks = 85
    side = rng.choice([-1.0, 1.0], size=n_cheeks)
    x_cheeks = side * rng.uniform(1.3, 4.3, size=n_cheeks)
    z_cheeks = rng.uniform(63.9, 65.3, size=n_cheeks)
    n_nose = 30
    x_nose = rng.normal(0.0, 0.40, size=n_nose)
    z_nose = rng.uniform(64.3, 65.4, size=n_nose)
    centers_x = np.concatenate([x_cheeks, x_nose])
    centers_z = np.concatenate([z_cheeks, z_nose])
    keep_mask = ~((np.abs(centers_x) < 1.1) & (centers_z < 63.85))
    centers_x = centers_x[keep_mask]
    centers_z = centers_z[keep_mask]

    face_mask = (
        (np.abs(verts[:, 0]) < 5.0)
        & (verts[:, 1] < -4.5)
        & (verts[:, 2] > 63.2)
        & (verts[:, 2] < 65.8)
        & (normals[:, 1] < -0.3)
    )
    if face_mask.any():
        fv = verts[face_mask]
        dx = fv[:, 0][:, None] - centers_x[None, :]
        dz = fv[:, 2][:, None] - centers_z[None, :]
        dist = np.sqrt((dx**2 + dz**2).min(axis=1))
        philtrum_mask = (np.abs(fv[:, 0]) < 1.0) & (fv[:, 2] < 63.85)
        weights = 0.28 * np.exp(-((dist / 0.11) ** 2))
        weights[philtrum_mask] = 0.0
        freckle = hex_srgb_to_linear(FRECKLE)
        colors[face_mask] = (
            colors[face_mask] * (1.0 - weights[:, None])
            + freckle[None, :] * weights[:, None]
        )
        painted["freckles"] = int((weights > 0.05).sum())

    colors[:, 3] = 1.0
    existing = mesh.color_attributes.get("SkinPaint")
    if existing:
        mesh.color_attributes.remove(existing)
    attr = mesh.color_attributes.new(name="SkinPaint", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", colors.ravel())

    rough[:, 3] = 1.0
    existing_rough = mesh.color_attributes.get("SkinRoughness")
    if existing_rough:
        mesh.color_attributes.remove(existing_rough)
    rough_attr = mesh.color_attributes.new(name="SkinRoughness", type="FLOAT_COLOR", domain="POINT")
    rough_attr.data.foreach_set("color", rough.ravel())
    mesh.color_attributes.active_color = attr
    mesh.update()
    return painted


def _noise(nodes, name, x, y, scale, detail, roughness):
    node = nodes.new("ShaderNodeTexNoise")
    node.name = name
    node.location = (x, y)
    set_input(node, "Scale", float(scale))
    set_input(node, "Detail", float(detail))
    set_input(node, "Roughness", float(roughness))
    return node


def _map_range(nodes, name, x, y, to_min, to_max):
    node = nodes.new("ShaderNodeMapRange")
    node.name = name
    node.location = (x, y)
    set_input(node, "From Min", 0.0)
    set_input(node, "From Max", 1.0)
    set_input(node, "To Min", float(to_min))
    set_input(node, "To Max", float(to_max))
    return node


def build_skin_material(obj: bpy.types.Object) -> bpy.types.Material:
    mat = bpy.data.materials.new("DewySatinPorcelain_8K")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    set_input(bsdf, "Metallic", 0.0)
    set_input(bsdf, "Specular IOR Level", 0.50)
    set_input(bsdf, "Specular Tint", (1.0, 0.96, 0.92, 1.0))
    set_input(bsdf, "Subsurface Weight", 0.46)
    set_input(bsdf, "Subsurface Radius", (2.2, 1.1, 0.65))
    set_input(bsdf, "Subsurface Scale", 0.05)
    set_input(bsdf, "Sheen Weight", 0.28)
    set_input(bsdf, "Sheen Roughness", 0.48)
    set_input(bsdf, "Sheen Tint", (1.0, 0.85, 0.78, 1.0))
    set_input(bsdf, "Coat Weight", 0.12)
    set_input(bsdf, "Coat Roughness", 0.32)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1100, 200)

    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = "SkinPaint"
    vcol.location = (-900, 350)

    coarse = _noise(nodes, "ToneNoise", -900, 0, 5.0, 2.0, 0.60)
    mid = _noise(nodes, "MottleNoise", -900, -125, 22.0, 2.0, 0.55)
    fine = _noise(nodes, "PoreNoise", -900, -250, 26.0, 2.0, 0.52)
    links.new(tex_coord.outputs["Object"], coarse.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], mid.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], fine.inputs["Vector"])

    coarse_range = _map_range(nodes, "ToneRange", -680, 0, 0.94, 1.01)
    mid_range = _map_range(nodes, "MottleRange", -680, -125, 0.95, 1.01)
    fine_range = _map_range(nodes, "PoreRange", -680, -250, 0.955, 1.0)
    links.new(mid.outputs["Fac"], mid_range.inputs["Value"])
    links.new(coarse.outputs["Fac"], coarse_range.inputs["Value"])
    links.new(fine.outputs["Fac"], fine_range.inputs["Value"])

    mix1 = nodes.new("ShaderNodeMixRGB")
    mix1.blend_type = "MULTIPLY"
    mix1.inputs["Fac"].default_value = 1.0
    mix1.location = (-460, 250)
    links.new(vcol.outputs["Color"], mix1.inputs["Color1"])
    links.new(coarse_range.outputs["Result"], mix1.inputs["Color2"])

    mix2 = nodes.new("ShaderNodeMixRGB")
    mix2.blend_type = "MULTIPLY"
    mix2.inputs["Fac"].default_value = 1.0
    mix2.location = (-260, 250)
    links.new(mix1.outputs["Color"], mix2.inputs["Color1"])
    links.new(fine_range.outputs["Result"], mix2.inputs["Color2"])

    mix3 = nodes.new("ShaderNodeMixRGB")
    mix3.blend_type = "MULTIPLY"
    mix3.inputs["Fac"].default_value = 1.0
    mix3.location = (-60, 250)
    links.new(mix2.outputs["Color"], mix3.inputs["Color1"])
    links.new(mid_range.outputs["Result"], mix3.inputs["Color2"])

    links.new(mix3.outputs["Color"], bsdf.inputs["Base Color"])
    if "Subsurface Color" in bsdf.inputs:
        links.new(mix3.outputs["Color"], bsdf.inputs["Subsurface Color"])

    # Roughness
    rough_vcol = nodes.new("ShaderNodeVertexColor")
    rough_vcol.layer_name = "SkinRoughness"
    rough_vcol.location = (-900, -650)
    rough_separate = nodes.new("ShaderNodeSeparateColor")
    rough_separate.location = (-710, -650)
    links.new(rough_vcol.outputs["Color"], rough_separate.inputs["Color"])

    rough_noise = _noise(nodes, "RoughNoise", -900, -500, 6.0, 2.0, 0.6)
    links.new(tex_coord.outputs["Object"], rough_noise.inputs["Vector"])
    rough_range = _map_range(nodes, "RoughRange", -680, -500, -0.035, 0.035)
    pore_rough_range = _map_range(nodes, "PoreRoughRange", -680, -625, -0.015, 0.015)
    links.new(rough_noise.outputs["Fac"], rough_range.inputs["Value"])
    links.new(fine.outputs["Fac"], pore_rough_range.inputs["Value"])

    rough_add1 = nodes.new("ShaderNodeMath")
    rough_add1.operation = "ADD"
    rough_add1.use_clamp = True
    rough_add1.location = (-440, -560)
    links.new(rough_separate.outputs["Red"], rough_add1.inputs[0])
    links.new(rough_range.outputs["Result"], rough_add1.inputs[1])

    rough_add2 = nodes.new("ShaderNodeMath")
    rough_add2.operation = "ADD"
    rough_add2.use_clamp = True
    rough_add2.location = (-240, -560)
    links.new(rough_add1.outputs["Value"], rough_add2.inputs[0])
    links.new(pore_rough_range.outputs["Result"], rough_add2.inputs[1])
    links.new(rough_add2.outputs["Value"], bsdf.inputs["Roughness"])

    # Micro-dermal bump in object space (completely UV-free and zero tangent artifacts)
    bump = nodes.new("ShaderNodeBump")
    bump.location = (-260, -150)
    set_input(bump, "Strength", 0.02)
    set_input(bump, "Distance", 0.05)
    links.new(fine.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def track_camera(obj, target: Vector) -> None:
    for c in list(obj.constraints):
        obj.constraints.remove(c)
    ct = obj.constraints.new(type="TRACK_TO")
    ct.target = bpy.data.objects.get("_TrackTarget")
    if ct.target is None:
        tt = bpy.data.objects.new("_TrackTarget", None)
        bpy.context.scene.collection.objects.link(tt)
        ct.target = tt
    ct.target.location = target
    ct.track_axis = "TRACK_NEGATIVE_Z"
    ct.up_axis = "UP_Y"


def add_light(name: str, location: tuple[float, float, float], energy: float, size: float, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    return obj


def main() -> None:
    args = parse_args()
    blend_path = Path(args.blend).resolve()
    out_path = Path(args.out).resolve()
    out_dir = Path(args.out_dir).resolve()
    preview_dir = out_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    log(f"=== Opening Original Blend File: {blend_path} ===")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    char_obj = bpy.data.objects.get("reVAmped_character_female_8k")
    if not char_obj:
        for o in bpy.data.objects:
            if o.type == "MESH" and "cyclorama" not in o.name.lower():
                char_obj = o
                break

    log(f"Target character mesh: '{char_obj.name}'")

    # Restore backdrop material to clean studio cyclorama
    cyc_obj = bpy.data.objects.get("Studio_Infinity_Cyclorama")
    if cyc_obj:
        cyc_mat = bpy.data.materials.get("M_Cinema_ObsidianSilk_Cyclorama")
        if not cyc_mat:
            cyc_mat = bpy.data.materials.new("M_Cinema_ObsidianSilk_Cyclorama")
            cyc_mat.use_nodes = True
            bsdf = cyc_mat.node_tree.nodes["Principled BSDF"]
            set_input(bsdf, "Base Color", (0.034, 0.030, 0.048, 1.0))
            set_input(bsdf, "Roughness", 0.58)
        cyc_obj.data.materials.clear()
        cyc_obj.data.materials.append(cyc_mat)
        log("Restored Studio_Infinity_Cyclorama backdrop material.")

    # Paint vertex colors (SkinPaint and SkinRoughness)
    log("Painting skin colors and roughness layers...")
    painted_stats = paint_skin_colors(char_obj)
    log(f"Painted skin stats: {painted_stats}")

    # Build the Dewy Satin Porcelain procedural material on the character
    log("Building Dewy Satin Porcelain skin material in Object Space...")
    skin_mat = build_skin_material(char_obj)
    log(f"Applied material '{skin_mat.name}' to '{char_obj.name}'.")

    # Save cleaned scene
    log(f"Saving fixed blend file to: {out_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)

    # Write report
    report_file = out_dir / args.report
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"Report written to: {report_file}")

    # Render Previews
    if args.render_previews:
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        scene.render.resolution_x = 1000
        scene.render.resolution_y = 1333

        # Active camera render
        if scene.camera is not None:
            preview_orig = preview_dir / "ICHxTenebra_preview.png"
            scene.render.filepath = str(preview_orig)
            log(f"Rendering active camera to {preview_orig}...")
            try:
                bpy.ops.render.render(write_still=True)
                log("Active camera render complete!")
            except Exception as e:
                log(f"Active camera render failed: {e}")

        # Studio Front and Closeup
        bbox = [char_obj.matrix_world @ Vector(c) for c in char_obj.bound_box]
        center = sum(bbox, Vector()) / 8.0
        size_z = max(v.z for v in bbox) - min(v.z for v in bbox)
        char_height = max(size_z, 1.0)

        cam_data = bpy.data.cameras.new("StudioCam")
        cam_obj = bpy.data.objects.new("StudioCam", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj

        add_light("StudioKey", (center.x - char_height * 0.5, center.y - char_height * 1.2, center.z + char_height * 0.8), 5000 * (char_height / 70.0)**2, char_height * 0.5, (1.0, 0.96, 0.92))
        add_light("StudioFill", (center.x + char_height * 0.6, center.y - char_height * 1.0, center.z + char_height * 0.4), 2000 * (char_height / 70.0)**2, char_height * 0.6, (0.94, 0.97, 1.0))
        add_light("StudioRim", (center.x, center.y + char_height * 1.0, center.z + char_height * 1.0), 4000 * (char_height / 70.0)**2, char_height * 0.5, (1.0, 0.9, 0.9))

        # Studio Front
        cam_obj.location = (center.x, center.y - char_height * 1.8, center.z + char_height * 0.1)
        track_camera(cam_obj, center)
        bpy.context.view_layer.update()

        front_path = preview_dir / "ICHxTenebra_front.png"
        scene.render.filepath = str(front_path)
        log(f"Rendering studio front to {front_path}...")
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            log(f"Front render failed: {e}")

        # Studio Closeup
        face_center = Vector((center.x, center.y, center.z + char_height * 0.35))
        cam_obj.location = (center.x + char_height * 0.15, center.y - char_height * 0.7, face_center.z + char_height * 0.05)
        track_camera(cam_obj, face_center)
        bpy.context.view_layer.update()

        closeup_path = preview_dir / "ICHxTenebra_closeup.png"
        scene.render.filepath = str(closeup_path)
        log(f"Rendering studio closeup to {closeup_path}...")
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            log(f"Closeup render failed: {e}")

    log("All repairs and renders completed!")


if __name__ == "__main__":
    main()
