"""Sculpt-smooth the joined Chibi model and give it a pretty skin texture.

Input is the joined STL (``Chibi_full.stl``) built by
``scripts/merge_chibi.py`` -- the full-detail body/head geometry *before*
the coarse fused remesh, so no fine original forms are lost.
The pipeline is:

1. re-voxel-remesh with a fine voxel size (higher, even quad density)
   which also fuses the joined shells into one watertight surface;
2. a gentle sculpt-style smooth pass (keeps the surface watertight);
3. paint soft skin tones into a vertex-colour layer: light peach skin,
   pink cheeks / nose tip / lips, faint warm fingertips, toes and knees;
4. build a procedural Cycles skin material (vertex colours + very fine
   pore noise + soft subsurface scattering);
5. Smart-UV-project and bake base-colour / roughness textures;
6. export a textured ``.blend`` / ``.glb`` and the smoother geometry as
   ``.stl``, plus soft Cycles preview renders.

Run with Blender, for example:

    blender --background --factory-startup -noaudio \
        --python scripts/enhance_chibi.py -- \
        --input Chibi_full.stl --out-dir . \
        --voxel-size 0.14 --texture-size 4096

Face landmarks below were measured on the placed head (face looks to -Y):
nose tip is the most forward point at about (0.1, -7.29, 63.9); the eye
sockets recede above Z ~= 65.5, so the cheeks sit near (+/-3.1, -6.1, 63.8).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # must be imported before bmesh when running via the bpy wheel
import bmesh  # noqa: F401  (kept for parity with merge_chibi)
import numpy as np
from mathutils import Vector

SCRIPT_DIR = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "scripts"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import merge_chibi as mc  # noqa: E402


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="Chibi_full.stl")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--name", default="Chibi_pretty")
    p.add_argument("--voxel-size", type=float, default=0.14)
    p.add_argument("--smooth-factor", type=float, default=0.25)
    p.add_argument("--smooth-iterations", type=int, default=2)
    p.add_argument("--texture-size", type=int, default=4096)
    p.add_argument("--bake-samples", type=int, default=8)
    p.add_argument("--render-samples", type=int, default=32)
    p.add_argument("--render-width", type=int, default=800)
    p.add_argument("--skip-render", action="store_true")
    args = p.parse_args(argv)
    if args.voxel_size <= 0:
        p.error("--voxel-size must be positive")
    if not 0.0 <= args.smooth_factor <= 1.0:
        p.error("--smooth-factor must be between 0 and 1")
    if args.texture_size < 256:
        p.error("--texture-size must be at least 256")
    return args


# --------------------------------------------------------------------------- #
# colour helpers
# --------------------------------------------------------------------------- #
def hex_srgb_to_linear(hex_color: str) -> np.ndarray:
    """'#FFDFC9' -> linear-space RGBA numpy vector."""
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.append(linear, 1.0)


def set_input(node, name, value) -> None:
    if name in node.inputs:
        node.inputs[name].default_value = value


# --------------------------------------------------------------------------- #
# skin painting (vertex colours)
# --------------------------------------------------------------------------- #
SKIN_HIGH = "#FFE1C8"  # head / upper body
SKIN_MAIN = "#FFD3B3"  # main fair peachy skin
SKIN_LOW = "#FFC5A0"   # slightly warmer legs
BLUSH = "#FF8FAB"
EYELIDS = "#FFB3C0"
NOSE = "#FF968F"
LIPS = "#F67A9C"
WARM = "#FFB79D"


def _blob(verts, normals, center, radii, strength, need_front=True):
    centre = np.asarray(center, dtype=np.float64)
    radius = np.asarray(radii, dtype=np.float64)
    d2 = np.sum(((verts - centre) / radius) ** 2, axis=1)
    weights = strength * np.exp(-2.2 * d2)
    if need_front:
        # the face looks towards -Y; do not wrap the blush onto the back
        frontness = np.clip((-normals[:, 1] - 0.35) / 0.40, 0.0, 1.0)
        weights *= frontness
    return np.clip(weights, 0.0, 1.0)


def paint_skin_colors(obj: bpy.types.Object) -> dict:
    mesh = obj.data
    verts = mc.vertices_np(mesh)
    normals = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)

    count = len(verts)
    colors = np.empty((count, 4), dtype=np.float64)

    # soft vertical gradient over the body
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

    painted = {}
    # cute blush: cheeks, nose tip, lips (measured face landmarks)
    painted["cheek_left"] = apply_blob((-3.15, -6.1, 63.8), (1.55, 2.2, 1.05), BLUSH, 0.72)
    painted["cheek_right"] = apply_blob((3.15, -6.1, 63.8), (1.55, 2.2, 1.05), BLUSH, 0.72)
    painted["eyelids"] = apply_blob((-2.4, -5.9, 66.0), (1.1, 1.8, 0.55), EYELIDS, 0.22)
    painted["eyelids"] += apply_blob((2.4, -5.9, 66.0), (1.1, 1.8, 0.55), EYELIDS, 0.22)
    painted["nose"] = apply_blob((0.05, -7.25, 63.95), (0.55, 0.9, 0.50), NOSE, 0.55)
    painted["lips"] = apply_blob((0.0, -6.75, 62.75), (0.55, 1.0, 0.35), LIPS, 0.48)
    # faint warm tint on hands, feet and knees
    painted["hands"] = apply_blob((-22.0, 0.0, 36.0), (2.5, 3.5, 4.0), WARM, 0.18, False)
    painted["hands"] += apply_blob((22.0, 0.0, 36.0), (2.5, 3.5, 4.0), WARM, 0.18, False)
    painted["knees"] = apply_blob((-7.5, -4.3, 17.0), (1.8, 1.5, 2.0), WARM, 0.15)
    painted["knees"] += apply_blob((7.5, -4.3, 17.0), (1.8, 1.5, 2.0), WARM, 0.15)
    feet_mask = verts[:, 2] < (z_min + 2.5)
    warm = hex_srgb_to_linear(WARM)
    colors[feet_mask] = colors[feet_mask] * 0.88 + warm[None, :] * 0.12
    painted["feet"] = int(feet_mask.sum())

    # cute freckles across the nose and cheeks: deterministic tiny dots,
    # painted per-vertex on the front-facing part of the face
    rng = np.random.default_rng(7)
    centers_x = np.clip(rng.normal(0.0, 2.1, 110), -4.6, 4.6)
    centers_z = np.clip(64.3 + rng.normal(0.0, 0.55, 110), 63.2, 65.4)
    face_mask = (
        (np.abs(verts[:, 0]) < 5.0)
        & (verts[:, 1] < -4.5)
        & (verts[:, 2] > 62.8)
        & (verts[:, 2] < 65.8)
        & (normals[:, 1] < -0.3)
    )
    if face_mask.any():
        fv = verts[face_mask]
        dx = fv[:, 0][:, None] - centers_x[None, :]
        dz = fv[:, 2][:, None] - centers_z[None, :]
        dist = np.sqrt((dx**2 + dz**2).min(axis=1))
        weights = 0.38 * np.exp(-((dist / 0.11) ** 2))
        freckle = hex_srgb_to_linear("#C98A5E")
        colors[face_mask] = (
            colors[face_mask] * (1.0 - weights[:, None])
            + freckle[None, :] * weights[:, None]
        )
        painted["freckles"] = int((weights > 0.05).sum())

    # slightly pinker fingertips
    tip_mask = (np.abs(verts[:, 0]) > 21.3) & (verts[:, 2] > 30.0) & (verts[:, 2] < 42.0)
    if tip_mask.any():
        tip = hex_srgb_to_linear("#FFC4D6")
        colors[tip_mask] = colors[tip_mask] * 0.85 + tip[None, :] * 0.15
        painted["fingertips"] = int(tip_mask.sum())

    colors[:, 3] = 1.0
    existing = mesh.color_attributes.get("SkinPaint")
    if existing:
        mesh.color_attributes.remove(existing)
    attr = mesh.color_attributes.new(name="SkinPaint", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", colors.ravel())
    mesh.color_attributes.active_color = attr
    mesh.update()
    print(f"SkinPaint vertex colours: {painted}")
    return painted


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
def _noise(nodes, name, x, y, scale, detail, roughness):
    node = nodes.new("ShaderNodeTexNoise")
    node.name = name
    node.location = (x, y)
    set_input(node, "Scale", scale)
    set_input(node, "Detail", detail)
    set_input(node, "Roughness", roughness)
    return node


def _map_range(nodes, name, x, y, to_min, to_max):
    node = nodes.new("ShaderNodeMapRange")
    node.name = name
    node.location = (x, y)
    try:
        node.clamp = True
    except Exception:
        pass
    set_input(node, "From Min", 0.0)
    set_input(node, "From Max", 1.0)
    set_input(node, "To Min", float(to_min))
    set_input(node, "To Max", float(to_max))
    return node


def _skin_bsdf_values(bsdf) -> None:
    set_input(bsdf, "Metallic", 0.0)
    set_input(bsdf, "Specular IOR Level", 0.35)
    set_input(bsdf, "Subsurface Weight", 0.35)
    set_input(bsdf, "Subsurface Radius", (2.4, 1.2, 0.7))
    set_input(bsdf, "Sheen Weight", 0.18)
    set_input(bsdf, "Sheen Roughness", 0.55)
    set_input(bsdf, "Sheen Tint", (1.0, 0.82, 0.72, 1.0))
    set_input(bsdf, "Coat Weight", 0.06)
    set_input(bsdf, "Coat Roughness", 0.5)


def build_source_material(obj: bpy.types.Object) -> bpy.types.Material:
    mat = bpy.data.materials.new("Skin_source")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    _skin_bsdf_values(bsdf)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1100, 200)

    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = "SkinPaint"
    vcol.location = (-900, 350)

    coarse = _noise(nodes, "ToneNoise", -900, 0, 6.0, 3.0, 0.65)
    fine = _noise(nodes, "PoreNoise", -900, -250, 160.0, 2.0, 0.55)
    links.new(tex_coord.outputs["Object"], coarse.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], fine.inputs["Vector"])

    coarse_range = _map_range(nodes, "ToneRange", -680, 0, 0.955, 1.0)
    fine_range = _map_range(nodes, "PoreRange", -680, -250, 0.93, 1.0)
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
    links.new(mix2.outputs["Color"], bsdf.inputs["Base Color"])

    rough_noise = _noise(nodes, "RoughNoise", -900, -500, 45.0, 2.0, 0.6)
    links.new(tex_coord.outputs["Object"], rough_noise.inputs["Vector"])
    rough_range = _map_range(nodes, "RoughRange", -680, -500, 0.42, 0.58)
    links.new(rough_noise.outputs["Fac"], rough_range.inputs["Value"])
    links.new(rough_range.outputs["Result"], bsdf.inputs["Roughness"])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (-260, -150)
    set_input(bump, "Strength", 0.012)
    set_input(bump, "Distance", 0.04)
    links.new(fine.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def bake_textures(obj, mat, tex_dir: Path, size: int, samples: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.render.bake.margin = 16
    scene.render.bake.use_clear = True
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.target = "IMAGE_TEXTURES"

    tex_dir.mkdir(parents=True, exist_ok=True)

    def make_image(name, width, colorspace, color):
        old = bpy.data.images.get(name)
        if old:
            bpy.data.images.remove(old)
        image = bpy.data.images.new(name, width=width, height=width, alpha=False)
        image.generated_color = color
        try:
            image.colorspace_settings.name = colorspace
        except Exception as exc:
            print(f"colorspace {colorspace} unavailable: {exc}")
        return image

    img_base = make_image("skin_basecolor", size, "sRGB", (1.0, 0.85, 0.75, 1.0))
    img_rough = make_image(
        "skin_roughness", max(512, size // 2), "Non-Color", (0.45, 0.45, 0.45, 1.0)
    )

    nodes = mat.node_tree.nodes

    def bake_into(image, bake_type, out_name):
        node = nodes.new("ShaderNodeTexImage")
        node.image = image
        node.location = (300, 300)
        for other in nodes:
            other.select = False
        node.select = True
        nodes.active = node
        mc.select_only([obj], active=obj)
        bpy.ops.object.mode_set(mode="OBJECT")
        print(f"Baking {bake_type} into {out_name} ...")
        bpy.ops.object.bake(type=bake_type)
        path = tex_dir / out_name
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        print(f"Saved {path} ({path.stat().st_size} bytes)")
        nodes.remove(node)
        return path

    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    base_path = bake_into(img_base, "DIFFUSE", "skin_basecolor.png")
    rough_path = bake_into(img_rough, "ROUGHNESS", "skin_roughness.png")
    return img_base, img_rough, base_path, rough_path


def build_final_material(obj, img_base, img_rough) -> bpy.types.Material:
    mat = bpy.data.materials.new("Skin_textured")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    _skin_bsdf_values(bsdf)

    base_node = nodes.new("ShaderNodeTexImage")
    base_node.image = img_base
    base_node.location = (-500, 250)
    base_node.label = "Skin base colour"

    rough_node = nodes.new("ShaderNodeTexImage")
    rough_node.image = img_rough
    rough_node.location = (-500, -50)
    rough_node.label = "Skin roughness"

    links.new(base_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])

    # keep the micro-pore bump in the Blender material (glTF keeps the maps)
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-750, -350)
    fine = _noise(nodes, "PoreNoiseFinal", -500, -350, 160.0, 2.0, 0.55)
    links.new(tex_coord.outputs["Object"], fine.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.location = (-260, -250)
    set_input(bump, "Strength", 0.012)
    set_input(bump, "Distance", 0.04)
    links.new(fine.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def smooth_and_remesh(obj, voxel_size: float, factor: float, iterations: int) -> None:
    mc.select_only([obj], active=obj)
    bpy.ops.object.mode_set(mode="OBJECT")

    obj.data.remesh_voxel_size = float(voxel_size)
    obj.data.remesh_voxel_adaptivity = 0.0
    print(f"Voxel remesh at {voxel_size} ...")
    bpy.ops.object.voxel_remesh()

    if iterations > 0 and factor > 0:
        mod = obj.modifiers.new("SculptSmooth", "SMOOTH")
        mod.factor = float(factor)
        mod.iterations = int(iterations)
        print(f"Sculpt smooth: factor={factor}, iterations={iterations}")
        bpy.ops.object.modifier_apply(modifier=mod.name)

    mc.ensure_outward_normals(obj)
    mesh = obj.data
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()


def smart_uv(obj) -> None:
    mc.select_only([obj], active=obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    print("Smart UV project ...")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66.0),
        island_margin=0.003,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.update()


# --------------------------------------------------------------------------- #
# pretty preview renders
# --------------------------------------------------------------------------- #
def _track(obj, target: Vector) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_area_light(name, location, target, energy, size, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    _track(obj, Vector(target))
    return obj


def render_previews(obj, preview_dir: Path, name: str, samples: int, width: int):
    scene = bpy.context.scene
    preview_dir.mkdir(parents=True, exist_ok=True)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = int(width * 4 / 3)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"

    world = bpy.data.worlds.get("PrettyWorld") or bpy.data.worlds.new("PrettyWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = tuple(
        hex_srgb_to_linear("#EFD9D4")
    )
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
    scene.world = world

    temp_objects = []
    bpy.ops.mesh.primitive_plane_add(size=260, location=(0, 10, -0.24))
    floor = bpy.context.view_layer.objects.active
    floor.name = "PreviewFloor"
    floor_mat = bpy.data.materials.new("PreviewFloor")
    floor_mat.use_nodes = True
    floor_bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    floor_bsdf.inputs["Base Color"].default_value = tuple(hex_srgb_to_linear("#F6E3DE"))
    floor_bsdf.inputs["Roughness"].default_value = 0.95
    floor.data.materials.append(floor_mat)
    temp_objects.append(floor)

    temp_objects.append(
        _add_area_light("KeyLight", (-24, -34, 66), (0, 0, 40), 21000, 34, (1.0, 0.96, 0.92))
    )
    temp_objects.append(
        _add_area_light("FillLight", (30, -26, 40), (0, 0, 38), 7500, 28, (0.94, 0.97, 1.0))
    )
    temp_objects.append(
        _add_area_light("RimLight", (4, 34, 78), (0, 0, 52), 16000, 26, (1.0, 0.9, 0.9))
    )

    cam_data = bpy.data.cameras.new("PreviewCamera")
    cam_data.type = "ORTHO"
    cam_data.clip_start = 0.1
    cam_data.clip_end = 600
    cam = bpy.data.objects.new("PreviewCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    temp_objects.append(cam)

    mc.select_only([obj], active=obj)
    shots = [
        ("front", (0, -78, 43), (0, 0, 38), 88.0),
        ("threequarter", (40, -64, 50), (0, 0, 38), 88.0),
        ("closeup", (9, -30, 68), (0, -1, 66), 25.0),
    ]
    written = []
    for suffix, location, target, ortho_scale in shots:
        cam.location = location
        cam_data.ortho_scale = ortho_scale
        _track(cam, Vector(target))
        bpy.context.view_layer.update()
        path = preview_dir / f"{name}_{suffix}.png"
        scene.render.filepath = str(path)
        print(f"Rendering {path} ...")
        bpy.ops.render.render(write_still=True)
        written.append(path.name)

    for temp in temp_objects:
        bpy.data.objects.remove(temp, do_unlink=True)
    scene.camera = None
    mc.select_only([obj], active=obj)
    return written


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    print(f"Blender {bpy.app.version_string}")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    obj = mc.load_binary_stl(input_path, args.name)
    obj.data.name = args.name
    mc.ensure_outward_normals(obj)
    input_stats = mc.mesh_report(obj)
    print(f"Input: {input_stats}")

    smooth_and_remesh(obj, args.voxel_size, args.smooth_factor, args.smooth_iterations)
    sculpted_stats = mc.mesh_report(obj)
    print(f"Sculpted: {sculpted_stats}")

    painted = paint_skin_colors(obj)
    smart_uv(obj)

    source_mat = build_source_material(obj)
    img_base, img_rough, base_path, rough_path = bake_textures(
        obj, source_mat, out_dir / "textures", args.texture_size, args.bake_samples
    )
    final_mat = build_final_material(obj, img_base, img_rough)
    if source_mat.name in bpy.data.materials:
        bpy.data.materials.remove(source_mat)
    for image in (img_base, img_rough):
        try:
            image.pack()
        except Exception as exc:
            print(f"packing {image.name} failed: {exc}")

    rendered = []
    if not args.skip_render:
        rendered = render_previews(
            obj, out_dir / "preview", args.name, args.render_samples, args.render_width
        )

    mc.remove_all_except({obj.name})
    mc.purge_orphans()
    bpy.context.view_layer.update()
    final_stats = mc.mesh_report(obj)
    print(f"Final: {final_stats}")

    outputs = {}
    blend_path = out_dir / f"{args.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
    outputs["blend"] = blend_path.name

    stl_path = out_dir / f"{args.name}.stl"
    mc.export_stl(obj, stl_path)
    outputs["stl"] = stl_path.name

    glb_path = out_dir / f"{args.name}.glb"
    mc.export_glb(obj, glb_path)
    outputs["glb"] = glb_path.name
    outputs["textures"] = [f"textures/{base_path.name}", f"textures/{rough_path.name}"]
    outputs["preview"] = [f"preview/{item}" for item in rendered]

    report = {
        "blender_version": bpy.app.version_string,
        "input": input_path.name,
        "parameters": {
            "voxel_size": args.voxel_size,
            "smooth_factor": args.smooth_factor,
            "smooth_iterations": args.smooth_iterations,
            "texture_size": args.texture_size,
            "bake_samples": args.bake_samples,
            "render_samples": args.render_samples,
        },
        "painted_vertex_counts": painted,
        "input_geometry": input_stats,
        "sculpted_geometry": sculpted_stats,
        "final_geometry": final_stats,
        "outputs": outputs,
    }
    (out_dir / f"{args.name}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("REPORT " + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
