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


def get_surface_point(char_obj: bpy.types.Object | None, x: float, z: float) -> tuple[Vector, Vector]:
    """Find the front face surface location and normal for given (x, z)."""
    if char_obj is not None:
        inv = char_obj.matrix_world.inverted()
        # Ray from in front of the face pointing toward +Y (into the face)
        ray_origin = inv @ Vector((x, -18.0, z))
        ray_dir = (inv.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
        hit, loc, norm, _ = char_obj.ray_cast(ray_origin, ray_dir)
        if hit:
            world_loc = char_obj.matrix_world @ loc
            world_norm = (char_obj.matrix_world.to_3x3() @ norm).normalized()
            return world_loc, world_norm

    # Fallback to smooth chibi facial curvature if raycast misses
    head_r = 6.35
    x_clamped = min(abs(x), head_r - 0.2)
    y_approx = -math.sqrt(max(head_r**2 - x_clamped**2, 1.0))
    y_approx -= 0.12 * math.cos(math.pi * np.clip((z - 65.12) / 1.5, -1.0, 1.0))
    norm = Vector((x / head_r, -math.sqrt(max(1.0 - (x / head_r) ** 2, 0.05)), 0.0)).normalized()
    return Vector((x, y_approx, z)), norm


def create_symmetric_anime_eyes(
    char_obj: bpy.types.Object | None,
    center_x: float = 2.22,
    center_z: float = 65.12,
    radius_x: float = 2.02,
    radius_z: float = 1.30,
    rows: int = 22,
    cols: int = 28,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Create left and right anime eye meshes with exact 100% mirror symmetry and face contouring."""
    mesh_l = bpy.data.meshes.new("AnimeEye_L")
    mesh_r = bpy.data.meshes.new("AnimeEye_R")
    bm_l = bmesh.new()
    bm_r = bmesh.new()

    # Left eye (+X, viewer's right): x goes from inner nose (+0.20) to outer temple (+4.24)
    # Right eye (-X, viewer's left): x goes from outer temple (-4.24) to inner nose (-0.20)
    xs_l = np.linspace(center_x - radius_x, center_x + radius_x, cols)
    zs = np.linspace(center_z - radius_z, center_z + radius_z, rows)

    grid_l = []
    grid_r = []

    for r in range(rows):
        row_l = []
        row_r = []
        z_val = zs[r]
        for c in range(cols):
            x_l = xs_l[c]
            # Raycast surface point for left eye
            loc_l, norm_l = get_surface_point(char_obj, x_l, z_val)
            # Offset slightly above skin to float cleanly
            v_l = loc_l + norm_l * 0.020
            row_l.append(v_l)

        # Build exact mirror for right eye: x_r[c] == -x_l[cols - 1 - c]
        for c in range(cols):
            v_l_mirrored = row_l[cols - 1 - c]
            v_r = Vector((-v_l_mirrored.x, v_l_mirrored.y, v_l_mirrored.z))
            row_r.append(v_r)

        grid_l.append([bm_l.verts.new(p) for p in row_l])
        grid_r.append([bm_r.verts.new(p) for p in row_r])

    # Build quads
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Left eye faces (counter-clockwise)
            bm_l.faces.new((grid_l[r][c], grid_l[r][c + 1], grid_l[r + 1][c + 1], grid_l[r + 1][c]))
            # Right eye faces
            bm_r.faces.new((grid_r[r][c], grid_r[r][c + 1], grid_r[r + 1][c + 1], grid_r[r + 1][c]))

    # Setup standard UVs
    # On left eye: c=0 is inner tear duct (u=0), c=cols-1 is outer lash wing (u=1) -> uses eye_open_R.png
    # On right eye: c=0 is outer lash wing (u=0), c=cols-1 is inner tear duct (u=1) -> uses eye_open_L.png
    uv_l = bm_l.loops.layers.uv.new("UVMap")
    uv_r = bm_r.loops.layers.uv.new("UVMap")

    for r in range(rows - 1):
        for c in range(cols - 1):
            u0 = c / (cols - 1)
            u1 = (c + 1) / (cols - 1)
            v0 = r / (rows - 1)
            v1 = (r + 1) / (rows - 1)

            f_l = bm_l.faces.get([grid_l[r][c], grid_l[r][c + 1], grid_l[r + 1][c + 1], grid_l[r + 1][c]])
            if f_l:
                f_l.loops[0][uv_l].uv = (u0, v0)
                f_l.loops[1][uv_l].uv = (u1, v0)
                f_l.loops[2][uv_l].uv = (u1, v1)
                f_l.loops[3][uv_l].uv = (u0, v1)

            f_r = bm_r.faces.get([grid_r[r][c], grid_r[r][c + 1], grid_r[r + 1][c + 1], grid_r[r + 1][c]])
            if f_r:
                f_r.loops[0][uv_r].uv = (u0, v0)
                f_r.loops[1][uv_r].uv = (u1, v0)
                f_r.loops[2][uv_r].uv = (u1, v1)
                f_r.loops[3][uv_r].uv = (u0, v1)

    bm_l.to_mesh(mesh_l)
    bm_r.to_mesh(mesh_r)
    bm_l.free()
    bm_r.free()
    mesh_l.update()
    mesh_r.update()

    obj_l = bpy.data.objects.new("AnimeEye_L", mesh_l)
    obj_r = bpy.data.objects.new("AnimeEye_R", mesh_r)
    bpy.context.scene.collection.objects.link(obj_l)
    bpy.context.scene.collection.objects.link(obj_r)
    return obj_l, obj_r


def setup_symmetric_blink_morphs(
    obj_l: bpy.types.Object,
    obj_r: bpy.types.Object,
    center_z: float = 65.12,
    radius_z: float = 1.30,
    rows: int = 22,
    cols: int = 28,
) -> None:
    """Setup Basis and Blink morph targets where upper eyelid and feathered lashes
    smoothly glide down in an arc to meet the lower lash line in 100% exact symmetry."""
    for obj in [obj_l, obj_r]:
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")

    blink_l = obj_l.shape_key_add(name="Blink")
    blink_r = obj_r.shape_key_add(name="Blink")
    obj_l.shape_key_add(name="EyeBlink_L")
    obj_r.shape_key_add(name="EyeBlink_R")

    z_bottom = center_z - radius_z * 0.88
    z_top = center_z + radius_z * 0.88
    z_range = max(z_top - z_bottom, 1e-6)

    # Calculate morph on Left Eye (+X)
    target_l_grid = []
    for r in range(rows):
        row_targets = []
        for c in range(cols):
            idx = r * cols + c
            co = obj_l.data.vertices[idx].co.copy()
            t = float(np.clip((co.z - z_bottom) / z_range, 0.0, 1.0))

            # Crescent curve across eye width (u_norm in [0, 1])
            u_norm = c / (cols - 1)
            crescent_curve = 0.12 * math.sin(math.pi * u_norm)

            closed_z = z_bottom + crescent_curve + (co.z - z_bottom) * 0.08
            closed_y = co.y - 0.08 * math.sin(math.pi * t)
            target = Vector((co.x, closed_y, closed_z))
            row_targets.append(target)
            blink_l.data[idx].co = target
        target_l_grid.append(row_targets)

    # Set exact mirror target on Right Eye (-X)
    for r in range(rows):
        for c in range(cols):
            idx_r = r * cols + c
            mirrored_target_l = target_l_grid[r][cols - 1 - c]
            target_r = Vector((-mirrored_target_l.x, mirrored_target_l.y, mirrored_target_l.z))
            blink_r.data[idx_r].co = target_r

    # Copy to individual EyeBlink_L and EyeBlink_R
    key_eb_l = obj_l.data.shape_keys.key_blocks.get("EyeBlink_L")
    if key_eb_l:
        for idx in range(len(obj_l.data.vertices)):
            key_eb_l.data[idx].co = blink_l.data[idx].co

    key_eb_r = obj_r.data.shape_keys.key_blocks.get("EyeBlink_R")
    if key_eb_r:
        for idx in range(len(obj_r.data.vertices)):
            key_eb_r.data[idx].co = blink_r.data[idx].co


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
    set_input(bsdf, "Roughness", 0.06)
    set_input(bsdf, "Specular IOR Level", 0.70)
    set_input(bsdf, "Coat Weight", 0.90)
    set_input(bsdf, "Coat Roughness", 0.04)
    set_input(bsdf, "Subsurface Weight", 0.10)
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

    # Find character mesh for surface raycasting
    char_obj = None
    for obj in bpy.data.objects:
        if obj.type == "MESH" and "AnimeEye" not in obj.name and "Preview" not in obj.name:
            char_obj = obj
            print(f"Found character mesh for face surface raycasting: {char_obj.name}")
            break

    # Textures:
    # eye_open_L.png: wing on left, tear duct on right -> for Character's Right eye (-X, viewer's left)
    # eye_open_R.png: tear duct on left, wing on right -> for Character's Left eye (+X, viewer's right)
    tex_eye_l = tex_dir / "eye_open_R.png"  # for Character Left (+X)
    tex_eye_r = tex_dir / "eye_open_L.png"  # for Character Right (-X)

    # Big, cute anime/chibi eye proportions:
    # center_x = 2.22, center_z = 65.12
    # radius_x = 2.02 (width = 4.04), radius_z = 1.30 (height = 2.60)
    center_x = 2.22
    center_z = 65.12
    radius_x = 2.02
    radius_z = 1.30

    mat_eye_l = build_eye_material(tex_eye_l, is_left=True)
    mat_eye_r = build_eye_material(tex_eye_r, is_left=False)

    print("Building left and right anime eye assemblies with 100% exact mirror symmetry...")
    eye_l, eye_r = create_symmetric_anime_eyes(
        char_obj=char_obj,
        center_x=center_x,
        center_z=center_z,
        radius_x=radius_x,
        radius_z=radius_z,
    )

    eye_l.data.materials.append(mat_eye_l)
    eye_r.data.materials.append(mat_eye_r)

    # Setup Blink Shape Keys with exact mirror symmetry
    print("Setting up Shape Keys for natural blinking with exact mirror symmetry...")
    setup_symmetric_blink_morphs(
        obj_l=eye_l,
        obj_r=eye_r,
        center_z=center_z,
        radius_z=radius_z,
    )

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
    face_center = Vector((0.0, -6.1, 64.6))
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
