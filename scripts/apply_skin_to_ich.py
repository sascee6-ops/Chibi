"""Apply Chibi pretty 8K Dewy Satin Porcelain skin material and textures to ICHxTenebra scene.

Can be run via Blender:
    blender --background --factory-startup -noaudio \
        --python scripts/apply_skin_to_ich.py -- \
        --blend "ICHxTenebra.v0.0.0.0.0.0-s0.blend" \
        --textures "textures" \
        --out "ICHxTenebra_with_skin.blend" \
        --render-previews
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="ICHxTenebra.v0.0.0.0.0.0-s0.blend")
    p.add_argument("--textures", default="textures")
    p.add_argument("--out", default="ICHxTenebra_with_skin.blend")
    p.add_argument("--render-previews", action="store_true")
    p.add_argument("--out-dir", default=".")
    return p.parse_args(argv)


def set_input(node, name: str, value) -> None:
    if name in node.inputs:
        node.inputs[name].default_value = value


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
    tex_dir = Path(args.textures).resolve()
    out_path = Path(args.out).resolve()
    out_dir = Path(args.out_dir).resolve()
    preview_dir = out_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading blend file: {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    print(f"Blender version: {bpy.app.version_string}")
    print(f"Objects in scene ({len(bpy.data.objects)}):")
    mesh_objects = []
    for obj in bpy.data.objects:
        print(f"  - {obj.name} (type={obj.type}, visible={not obj.hide_get()})")
        if obj.type == "MESH":
            mesh_objects.append(obj)

    print(f"Materials in scene ({len(bpy.data.materials)}):")
    for mat in bpy.data.materials:
        print(f"  - {mat.name}")

    # Locate the 8K skin textures
    basecolor_path = next(tex_dir.glob("*basecolor*.png"), None)
    roughness_path = next(tex_dir.glob("*roughness*.png"), None)
    normal_path = next(tex_dir.glob("*normal*.png"), None)

    if not basecolor_path or not basecolor_path.exists():
        for p in Path(".").rglob("*skin_basecolor*.png"):
            basecolor_path = p.resolve()
            break
    if not roughness_path or not roughness_path.exists():
        for p in Path(".").rglob("*skin_roughness*.png"):
            roughness_path = p.resolve()
            break
    if not normal_path or not normal_path.exists():
        for p in Path(".").rglob("*skin_normal*.png"):
            normal_path = p.resolve()
            break

    print(f"BaseColor: {basecolor_path}")
    print(f"Roughness: {roughness_path}")
    print(f"Normal: {normal_path}")

    # Load images into Blender
    img_base = bpy.data.images.load(str(basecolor_path), check_existing=True) if basecolor_path and basecolor_path.exists() else None
    img_rough = bpy.data.images.load(str(roughness_path), check_existing=True) if roughness_path and roughness_path.exists() else None
    img_norm = bpy.data.images.load(str(normal_path), check_existing=True) if normal_path and normal_path.exists() else None

    if img_base:
        img_base.colorspace_settings.name = "sRGB"
    if img_rough:
        img_rough.colorspace_settings.name = "Non-Color"
    if img_norm:
        img_norm.colorspace_settings.name = "Non-Color"

    # Create the Dewy Satin Porcelain skin material
    skin_mat = bpy.data.materials.new("DewySatinPorcelain_8K")
    skin_mat.use_nodes = True
    nodes = skin_mat.node_tree.nodes
    links = skin_mat.node_tree.links
    nodes.clear()

    # Output node
    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    # Configure Dewy Satin Porcelain SSS and Clearcoat
    set_input(bsdf, "Metallic", 0.0)
    set_input(bsdf, "Roughness", 0.46)
    set_input(bsdf, "Subsurface Weight", 0.46)
    set_input(bsdf, "Subsurface Radius", (2.2, 1.1, 0.65))
    set_input(bsdf, "Subsurface Scale", 0.05)
    set_input(bsdf, "Sheen Weight", 0.28)
    set_input(bsdf, "Sheen Tint", (1.0, 0.88, 0.85, 1.0))
    set_input(bsdf, "Coat Weight", 0.12)
    set_input(bsdf, "Coat Roughness", 0.08)

    # Base Color texture
    if img_base:
        tex_base = nodes.new("ShaderNodeTexImage")
        tex_base.image = img_base
        tex_base.location = (-400, 200)
        links.new(tex_base.outputs["Color"], bsdf.inputs["Base Color"])
        if "Subsurface Color" in bsdf.inputs:
            links.new(tex_base.outputs["Color"], bsdf.inputs["Subsurface Color"])

    # Roughness texture
    if img_rough:
        tex_rough = nodes.new("ShaderNodeTexImage")
        tex_rough.image = img_rough
        tex_rough.location = (-400, -50)
        links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])

    # Normal map
    if img_norm:
        tex_norm = nodes.new("ShaderNodeTexImage")
        tex_norm.image = img_norm
        tex_norm.location = (-600, -300)
        norm_map = nodes.new("ShaderNodeNormalMap")
        norm_map.location = (-300, -300)
        set_input(norm_map, "Strength", 0.58)
        links.new(tex_norm.outputs["Color"], norm_map.inputs["Color"])
        links.new(norm_map.outputs["Normal"], bsdf.inputs["Normal"])

    # Apply skin material to mesh objects in the scene
    applied_count = 0
    target_mesh = None
    for obj in mesh_objects:
        name_lower = obj.name.lower()
        is_body_target = any(k in name_lower for k in ["body", "skin", "head", "face", "torso", "mesh", "character", "ich", "chibi"]) or len(obj.data.materials) == 0
        if is_body_target:
            target_mesh = obj
            print(f"Applying Dewy Satin Porcelain skin material to mesh: {obj.name}")
            if len(obj.data.materials) == 0:
                obj.data.materials.append(skin_mat)
            else:
                replaced = False
                for i, slot in enumerate(obj.material_slots):
                    slot_name = (slot.material.name if slot.material else "").lower()
                    if any(k in slot_name for k in ["skin", "body", "face", "head", "flesh", "porcelain"]):
                        slot.material = skin_mat
                        replaced = True
                        print(f"  Replaced material slot {i} ('{slot_name}') with DewySatinPorcelain_8K")
                if not replaced:
                    obj.data.materials[0] = skin_mat
                    print(f"  Set material slot 0 on {obj.name} to DewySatinPorcelain_8K")
            applied_count += 1

    print(f"Applied skin material to {applied_count} meshes.")

    # Pack textures into the blend file
    for img in [img_base, img_rough, img_norm]:
        if img:
            try:
                img.pack()
                print(f"Packed texture: {img.name}")
            except Exception as e:
                print(f"Could not pack {img.name}: {e}")

    # Save the updated blend file
    print(f"Saving updated scene to: {out_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)

    # Render previews
    if args.render_previews:
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        scene.render.resolution_x = 1000
        scene.render.resolution_y = 1333

        # 1. Render default scene camera if present
        if scene.camera is not None:
            preview_orig = out_dir / "preview_ICH_skin.png"
            scene.render.filepath = str(preview_orig)
            print(f"Rendering active scene camera to: {preview_orig}")
            try:
                bpy.ops.render.render(write_still=True)
                # Also copy to preview/
                import shutil
                shutil.copyfile(preview_orig, preview_dir / "preview_ICH_skin.png")
                shutil.copyfile(preview_orig, preview_dir / "ICHxTenebra_preview.png")
            except Exception as e:
                print(f"Rendering default camera failed: {e}")

        # 2. Setup studio camera & lights focused on the mesh
        ref_obj = target_mesh or (mesh_objects[0] if mesh_objects else None)
        if ref_obj:
            bbox = [ref_obj.matrix_world @ Vector(corner) for corner in ref_obj.bound_box]
            center = sum(bbox, Vector()) / 8.0
            size_z = max(v.z for v in bbox) - min(v.z for v in bbox)
            size_y = max(v.y for v in bbox) - min(v.y for v in bbox)
            size_x = max(v.x for v in bbox) - min(v.x for v in bbox)
            char_height = max(size_z, 1.0)
            print(f"Character bounding box center={center}, height={char_height}")

            # Camera
            cam_data = bpy.data.cameras.new("StudioCam")
            cam_obj = bpy.data.objects.new("StudioCam", cam_data)
            scene.collection.objects.link(cam_obj)
            scene.camera = cam_obj

            # Key, Fill, Rim lights
            l1 = add_light("StudioKey", (center.x - char_height * 0.5, center.y - char_height * 1.2, center.z + char_height * 0.8), 5000 * (char_height / 70.0)**2, char_height * 0.5, (1.0, 0.96, 0.92))
            l2 = add_light("StudioFill", (center.x + char_height * 0.6, center.y - char_height * 1.0, center.z + char_height * 0.4), 2000 * (char_height / 70.0)**2, char_height * 0.6, (0.94, 0.97, 1.0))
            l3 = add_light("StudioRim", (center.x, center.y + char_height * 1.0, center.z + char_height * 1.0), 4000 * (char_height / 70.0)**2, char_height * 0.5, (1.0, 0.9, 0.9))

            # Studio Front render
            cam_obj.location = (center.x, center.y - char_height * 1.8, center.z + char_height * 0.1)
            track_camera(cam_obj, center)
            bpy.context.view_layer.update()

            front_path = preview_dir / "ICHxTenebra_front.png"
            scene.render.filepath = str(front_path)
            print(f"Rendering studio front to {front_path}")
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"Front render failed: {e}")

            # Studio Closeup render (focused on face/upper body)
            face_center = Vector((center.x, center.y, center.z + char_height * 0.35))
            cam_obj.location = (center.x + char_height * 0.15, center.y - char_height * 0.7, face_center.z + char_height * 0.05)
            track_camera(cam_obj, face_center)
            bpy.context.view_layer.update()

            closeup_path = preview_dir / "ICHxTenebra_closeup.png"
            scene.render.filepath = str(closeup_path)
            print(f"Rendering studio closeup to {closeup_path}")
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"Closeup render failed: {e}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
