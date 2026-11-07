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
from mathutils import Vector, Euler, Matrix


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


def create_eyeball_mesh(name: str, center: Vector, radius: float, is_left: bool) -> bpy.types.Object:
    """Create a curved anime eyeball surface (sclera + iris plane) with smooth normals."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    # Generate curved grid dome for eyeball
    rows, cols = 16, 20
    # Angular span across socket
    x_span = math.radians(65.0)
    z_span = math.radians(55.0)

    grid_verts = []
    sign = 1.0 if is_left else -1.0

    for r in range(rows):
        row_verts = []
        v_frac = r / (rows - 1)
        phi = (v_frac - 0.5) * z_span
        for c in range(cols):
            u_frac = c / (cols - 1)
            theta = (u_frac - 0.5) * x_span

            # Spherical projection facing -Y
            x = center.x + radius * math.sin(theta)
            y = center.y - radius * math.cos(theta) * math.cos(phi) + radius * 0.85
            z = center.z + radius * math.sin(phi)

            vert = bm.verts.new((x, y, z))
            row_verts.append(vert)
        grid_verts.append(row_verts)

    # Faces
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
            # Map U, V cleanly across [0, 1]
            u0 = c / (cols - 1)
            u1 = (c + 1) / (cols - 1)
            v0 = r / (rows - 1)
            v1 = (r + 1) / (rows - 1)
            if not is_left:
                u0, u1 = 1.0 - u0, 1.0 - u1

            # Match face loop order
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


def create_feathered_lash_mesh(name: str, center: Vector, radius: float, is_left: bool) -> bpy.types.Object:
    """Create stylized feathered white eyelashes with distinct spiky tufts and curvature."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    sign = 1.0 if is_left else -1.0
    num_spikes = 7
    # Arc along upper eyelid
    arc_angles = np.linspace(-math.radians(35), math.radians(40), num_spikes)
    if not is_left:
        arc_angles = -arc_angles

    # Base curve along the lid
    for i, angle in enumerate(arc_angles):
        # Base on eyelid rim
        bx = center.x + sign * (radius * 1.05 * math.sin(angle))
        by = center.y - radius * 0.95 * math.cos(angle) + radius * 0.82
        bz = center.z + radius * 0.45 + (1.0 - (i / num_spikes - 0.5)**2) * 0.35

        # Tapering spike tip sweeping outward and upward
        tip_length = 0.55 + 0.45 * (i / num_spikes)
        tx = bx + sign * (0.35 + 0.25 * i)
        ty = by - 0.15 - 0.1 * math.sin(angle)
        tz = bz + tip_length * (0.4 + 0.6 * math.sin(abs(angle) + 0.3))

        # Build triangular/quad feather tuft
        v_base1 = bm.verts.new((bx - 0.15 * sign, by, bz - 0.08))
        v_base2 = bm.verts.new((bx + 0.15 * sign, by, bz - 0.08))
        v_mid1  = bm.verts.new((bx * 0.6 + tx * 0.4 - 0.08 * sign, by * 0.6 + ty * 0.4, bz * 0.6 + tz * 0.4))
        v_mid2  = bm.verts.new((bx * 0.6 + tx * 0.4 + 0.08 * sign, by * 0.6 + ty * 0.4, bz * 0.6 + tz * 0.4))
        v_tip   = bm.verts.new((tx, ty, tz))

        bm.faces.new((v_base1, v_base2, v_mid2, v_mid1))
        bm.faces.new((v_mid1, v_mid2, v_tip))

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_blink_shape_keys(obj: bpy.types.Object, is_eyelash: bool, center: Vector) -> None:
    """Create Basis and Blink shape keys with correct smooth downward arc rotation."""
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")

    blink_key = obj.shape_key_add(name="Blink")
    blink_l_key = obj.shape_key_add(name="EyeBlink_L")
    blink_r_key = obj.shape_key_add(name="EyeBlink_R")

    # Deform vertices for closed eye state
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    delta_z = -1.15 if is_eyelash else -0.75
    for vert in bm.verts:
        # Distance from eyelid bottom
        dz = vert.co.z - center.z
        if is_eyelash:
            # Upper eyelashes slide down over the eye and fan outward/downward
            vert.co.z += delta_z
            vert.co.y += 0.08  # glide forward over eyeball curve
        else:
            # Eyelid skin covers the pupil
            if dz > 0:
                vert.co.z += delta_z * (dz / 1.0)
                vert.co.y += 0.05

    bm.to_mesh(obj.data)
    # Store coordinates into shape key
    for idx, vert in enumerate(bm.verts):
        blink_key.data[idx].co = vert.co
        blink_l_key.data[idx].co = vert.co
        blink_r_key.data[idx].co = vert.co

    bm.free()
    # Reset base mesh coordinates back to Basis
    obj.data.update()


def build_eye_material(tex_path: Path, is_eyelash=False) -> bpy.types.Material:
    """Build Cycles AgX material with glowing amber iris or feathered white lashes."""
    mat_name = "M_AnimeEye_FeatherLash" if is_eyelash else "M_AnimeEye_Iris"
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else None
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    if is_eyelash:
        # Pure shimmering white / silver feather lashes
        set_input(bsdf, "Base Color", (0.96, 0.96, 0.98, 1.0))
        set_input(bsdf, "Roughness", 0.35)
        set_input(bsdf, "Sheen Weight", 0.65)
        set_input(bsdf, "Sheen Tint", (1.0, 1.0, 1.0, 1.0))
        set_input(bsdf, "Coat Weight", 0.40)
        set_input(bsdf, "Coat Roughness", 0.15)
        set_input(bsdf, "Subsurface Weight", 0.25)
        set_input(bsdf, "Subsurface Radius", (1.0, 1.0, 1.0))
    else:
        # Amber luminous anime eye iris
        set_input(bsdf, "Roughness", 0.08)
        set_input(bsdf, "Specular IOR Level", 0.65)
        set_input(bsdf, "Coat Weight", 0.85)
        set_input(bsdf, "Coat Roughness", 0.05)
        set_input(bsdf, "Subsurface Weight", 0.15)
        set_input(bsdf, "Subsurface Radius", (1.8, 1.0, 0.5))

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

    print(f"Loading scene: {blend_path}")
    if blend_path.exists():
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    else:
        print("Starting with empty scene...")
        bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene

    # Textures
    tex_eye_l = tex_dir / "eye_open_L.png"
    tex_eye_r = tex_dir / "eye_open_R.png"
    tex_iris = tex_dir / "eye_iris.png"
    tex_closed = tex_dir / "eye_closed.png"

    # Coordinates measured on head
    # Left eye: (+2.45, -6.05, 65.95)
    # Right eye: (-2.45, -6.05, 65.95)
    eye_radius = 1.35
    pos_l = Vector((2.45, -6.05, 65.95))
    pos_r = Vector((-2.45, -6.05, 65.95))

    # Materials
    mat_eye_l = build_eye_material(tex_eye_l)
    mat_eye_r = build_eye_material(tex_eye_r)
    mat_lash = build_eye_material(tex_closed, is_eyelash=True)

    print("Building left and right eye meshes...")
    eyeball_l = create_eyeball_mesh("AnimeEye_L", pos_l, eye_radius, is_left=True)
    eyeball_r = create_eyeball_mesh("AnimeEye_R", pos_r, eye_radius, is_left=False)

    eyeball_l.data.materials.append(mat_eye_l)
    eyeball_r.data.materials.append(mat_eye_r)

    print("Building feathered white eyelash geometry...")
    lash_l = create_feathered_lash_mesh("FeatherLash_Upper_L", pos_l, eye_radius, is_left=True)
    lash_r = create_feathered_lash_mesh("FeatherLash_Upper_R", pos_r, eye_radius, is_left=False)

    lash_l.data.materials.append(mat_lash)
    lash_r.data.materials.append(mat_lash)

    # Setup Blink Shape Keys
    print("Setting up Shape Keys for natural blinking...")
    for obj in [eyeball_l, eyeball_r]:
        setup_blink_shape_keys(obj, is_eyelash=False, center=pos_l if "L" in obj.name else pos_r)
    for obj in [lash_l, lash_r]:
        setup_blink_shape_keys(obj, is_eyelash=True, center=pos_l if "L" in obj.name else pos_r)

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
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb_path),
        export_format="GLB",
        export_morph=True,
        export_morph_normal=True,
        export_materials="EXPORT",
    )

    # Studio Lighting & Camera for Renders
    face_center = Vector((0.0, -6.1, 65.5))
    cam_data = bpy.data.cameras.new("FacePreviewCam")
    cam_obj = bpy.data.objects.new("FacePreviewCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Soft studio lights
    add_light("EyeKey", (-15.0, -30.0, 75.0), 12000, 20.0, (1.0, 0.96, 0.92))
    add_light("EyeFill", (18.0, -25.0, 68.0), 6000, 25.0, (0.95, 0.98, 1.0))
    add_light("EyeRim", (0.0, 25.0, 80.0), 9000, 20.0, (1.0, 0.90, 0.90))

    # Closeup framing
    cam_obj.location = (0.0, -22.0, 66.2)
    track_camera(cam_obj, face_center)
    cam_data.lens = 85.0  # 85mm portrait lens for flattering anime proportions
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
    for obj in [eyeball_l, eyeball_r, lash_l, lash_r]:
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
    for obj in [eyeball_l, eyeball_r, lash_l, lash_r]:
        if obj.data.shape_keys:
            key = obj.data.shape_keys.key_blocks.get("Blink")
            if key:
                key.value = 0.0

    print("All eye creation, morph targets, and renders finished successfully!")


if __name__ == "__main__":
    main()
