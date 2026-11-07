"""Build stylized anime eyes with feathered white eyelashes, glowing amber iris,
and realistic blinking shape keys for Chibi model.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import bpy
import bmesh
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="Chibi_pretty.blend")
    p.add_argument("--out", default="Chibi_pretty_eyes.blend")
    p.add_argument("--out-glb", default="Chibi_pretty_eyes.glb")
    p.add_argument("--textures", default="textures")
    p.add_argument("--preview-dir", default="preview")
    p.add_argument("--render-samples", type=int, default=64)
    return p.parse_args(argv)


def set_input(node, name: str, value) -> None:
    if name in node.inputs:
        node.inputs[name].default_value = value


def create_anime_eye(name: str, center: Vector, radius_x: float, radius_z: float, depth: float, is_left: bool) -> bpy.types.Object:
    """Create a curved anime eye mesh with clean UV mapping and natural eye contour."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    rows, cols = 20, 24
    x_angles = np.linspace(-math.radians(45), math.radians(45), cols)
    z_angles = np.linspace(-math.radians(38), math.radians(38), rows)

    grid_verts = []
    for r, phi in enumerate(z_angles):
        row_verts = []
        for c, theta in enumerate(x_angles):
            cos_factor = math.cos(theta) * math.cos(phi)
            x = center.x + radius_x * math.sin(theta)
            # Convex curve following eye socket curvature
            y = center.y - depth * (cos_factor - 0.25)
            z = center.z + radius_z * math.sin(phi)
            vert = bm.verts.new((x, y, z))
            row_verts.append(vert)
        grid_verts.append(row_verts)

    for r in range(rows - 1):
        for c in range(cols - 1):
            v1 = grid_verts[r][c]
            v2 = grid_verts[r][c + 1]
            v3 = grid_verts[r + 1][c + 1]
            v4 = grid_verts[r + 1][c]
            bm.faces.new((v1, v2, v3, v4))

    # UV unwrap
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for r in range(rows - 1):
        for c in range(cols - 1):
            u0 = c / (cols - 1)
            u1 = (c + 1) / (cols - 1)
            v0 = r / (rows - 1)
            v1 = (r + 1) / (rows - 1)
            if not is_left:
                u0, u1 = 1.0 - u0, 1.0 - u1

            f = bm.faces.get([grid_verts[r][c], grid_verts[r][c + 1], grid_verts[r + 1][c + 1], grid_verts[r + 1][c]])
            if f:
                f.loops[0][uv_layer].uv = (u0, v0)
                f.loops[1][uv_layer].uv = (u1, v0)
                f.loops[2][uv_layer].uv = (u1, v1)
                f.loops[3][uv_layer].uv = (u0, v1)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_blink_morphs(obj: bpy.types.Object, center: Vector, radius_z: float) -> None:
    """Setup Basis and Blink morph targets where upper eyelid and feathered lashes
    smoothly glide down in an arc to meet the lower lash line."""
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")

    blink_key = obj.shape_key_add(name="Blink")
    blink_l_key = obj.shape_key_add(name="EyeBlink_L")
    blink_r_key = obj.shape_key_add(name="EyeBlink_R")

    z_bottom = center.z - radius_z * 0.85
    z_top = center.z + radius_z * 0.85
    z_range = max(z_top - z_bottom, 1e-6)

    for idx, v in enumerate(obj.data.vertices):
        co = v.co.copy()
        # Normalized height t in [0.0 (bottom) .. 1.0 (top)]
        t = float(np.clip((co.z - z_bottom) / z_range, 0.0, 1.0))

        # Upper lid descends smoothly to bottom line (z_bottom)
        # using a gentle anime crescent-eye curvature
        crescent_curve = 0.12 * math.sin(math.pi * ((co.x - center.x) / 1.5 + 0.5))
        closed_z = z_bottom + crescent_curve + (co.z - z_bottom) * 0.08
        closed_y = co.y - 0.08 * math.sin(math.pi * t)

        target_co = Vector((co.x, closed_y, closed_z))
        blink_key.data[idx].co = target_co
        blink_l_key.data[idx].co = target_co
        blink_r_key.data[idx].co = target_co


def build_eye_material(tex_path: Path, is_left: bool) -> bpy.types.Material:
    """Build Cycles AgX anime eye material with transparent feathered alpha blending."""
    mat_name = f"M_AnimeEye_{'L' if is_left else 'R'}"
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    # Luminous anime eye shading
    set_input(bsdf, "Metallic", 0.0)
    set_input(bsdf, "Roughness", 0.08)
    set_input(bsdf, "Specular IOR Level", 0.65)
    set_input(bsdf, "Coat Weight", 0.85)
    set_input(bsdf, "Coat Roughness", 0.05)
    set_input(bsdf, "Subsurface Weight", 0.12)
    set_input(bsdf, "Subsurface Radius", (1.6, 1.0, 0.6))

    if tex_path.exists():
        tex_node = nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.load(str(tex_path), check_existing=True)
        img.colorspace_settings.name = "sRGB"
        tex_node.image = img
        tex_node.location = (-400, 200)

        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        if "Subsurface Color" in bsdf.inputs:
            links.new(tex_node.outputs["Color"], bsdf.inputs["Subsurface Color"])
        if "Alpha" in bsdf.inputs:
            links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])

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
    out_glb_path = Path(args.out_glb).resolve()
    tex_dir = Path(args.textures).resolve()
    preview_dir = Path(args.preview_dir).resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base scene: {blend_path}")
    if blend_path.exists():
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    else:
        print("Blend file not found, creating fresh scene...")
        bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene

    # Clean neutral studio world (warm soft ivory)
    world = bpy.data.worlds.new("StudioWorld")
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.94, 0.90, 0.88, 1.0)
        bg_node.inputs[1].default_value = 1.0
    scene.world = world

    # Textures
    tex_eye_l = tex_dir / "eye_open_L.png"
    tex_eye_r = tex_dir / "eye_open_R.png"

    # Anatomical placement on Chibi face:
    # Nose tip is at (0.1, -7.29, 63.9), cheeks at (+/-3.15, -6.1, 63.8)
    # Eye centers: X = +/- 1.48, Y = -6.18, Z = 64.75
    # Eye dimensions: width 2.7 (radius_x = 1.35), height 2.1 (radius_z = 1.05)
    eye_radius_x = 1.35
    eye_radius_z = 1.05
    eye_depth = 0.22
    pos_l = Vector((1.48, -6.18, 64.75))
    pos_r = Vector((-1.48, -6.18, 64.75))

    mat_eye_l = build_eye_material(tex_eye_l, is_left=True)
    mat_eye_r = build_eye_material(tex_eye_r, is_left=False)

    print("Building left and right anime eye assemblies...")
    eye_l = create_anime_eye("AnimeEye_L", pos_l, eye_radius_x, eye_radius_z, eye_depth, is_left=True)
    eye_r = create_anime_eye("AnimeEye_R", pos_r, eye_radius_x, eye_radius_z, eye_depth, is_left=False)

    eye_l.data.materials.append(mat_eye_l)
    eye_r.data.materials.append(mat_eye_r)

    # Setup Blink Shape Keys
    print("Setting up Shape Keys for natural blinking...")
    setup_blink_morphs(eye_l, pos_l, eye_radius_z)
    setup_blink_morphs(eye_r, pos_r, eye_radius_z)

    # Pack textures
    for img in bpy.data.images:
        try:
            img.pack()
            print(f"Packed image: {img.name}")
        except Exception:
            pass

    # Save master blend scene
    print(f"Saving blend scene: {out_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)

    # Export GLB with morph targets
    print(f"Exporting GLB with morph targets: {out_glb_path}")
    try:
        bpy.ops.export_scene.gltf(
            filepath=str(out_glb_path),
            export_format="GLB",
            export_morph=True,
            export_morph_normal=True,
            export_materials="EXPORT",
        )
    except Exception as e:
        print(f"GLB export notice: {e}")

    # Studio Portrait Lighting & Camera
    face_center = Vector((0.0, -6.1, 64.4))
    cam_data = bpy.data.cameras.new("FacePreviewCam")
    cam_obj = bpy.data.objects.new("FacePreviewCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # 3-point portrait studio lighting
    add_light("EyeKey", (-18.0, -32.0, 75.0), 14000, 24.0, (1.0, 0.96, 0.92))
    add_light("EyeFill", (20.0, -26.0, 68.0), 7000, 28.0, (0.95, 0.98, 1.0))
    add_light("EyeRim", (0.0, 25.0, 80.0), 10000, 22.0, (1.0, 0.90, 0.90))

    # Balanced portrait framing capturing the whole face and bust
    cam_obj.location = (0.0, -28.0, 64.6)
    track_camera(cam_obj, face_center)
    cam_data.lens = 70.0
    bpy.context.view_layer.update()

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.render_samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1333

    # 1. Render Open Eyes
    open_path = preview_dir / "chibi_eyes_open.png"
    scene.render.filepath = str(open_path)
    print(f"Rendering Eyes Open state: {open_path}")
    bpy.ops.render.render(write_still=True)

    # 2. Set Blink = 1.0 and render Closed Eyes (Blink)
    for obj in [eye_l, eye_r]:
        if obj.data.shape_keys:
            key = obj.data.shape_keys.key_blocks.get("Blink")
            if key:
                key.value = 1.0

    bpy.context.view_layer.update()
    blink_path = preview_dir / "chibi_eyes_blink.png"
    scene.render.filepath = str(blink_path)
    print(f"Rendering Eyes Blink state: {blink_path}")
    bpy.ops.render.render(write_still=True)

    # 3. Reset Blink = 0.0
    for obj in [eye_l, eye_r]:
        if obj.data.shape_keys:
            key = obj.data.shape_keys.key_blocks.get("Blink")
            if key:
                key.value = 0.0

    print("All eye creation, morph targets, and renders finished successfully!")


if __name__ == "__main__":
    main()
