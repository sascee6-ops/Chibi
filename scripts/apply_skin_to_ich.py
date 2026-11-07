"""Apply Chibi pretty 8K Dewy Satin Porcelain skin material and textures to ICHxTenebra scene.

Can be run via Blender:
    blender --background --factory-startup -noaudio \
        --python scripts/apply_skin_to_ich.py -- \
        --blend "ICHxTenebra.v0.0.0.0.0.0-s0.blend" \
        --textures "textures" \
        --out "ICHxTenebra_with_skin.blend"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import bpy


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


def main() -> None:
    args = parse_args()
    blend_path = Path(args.blend).resolve()
    tex_dir = Path(args.textures).resolve()
    out_path = Path(args.out).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading blend file: {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    print(f"Blender version: {bpy.app.version_string}")
    print(f"Objects in scene ({len(bpy.data.objects)}):")
    for obj in bpy.data.objects:
        print(f"  - {obj.name} (type={obj.type}, visible={not obj.hide_get()})")

    print(f"Materials in scene ({len(bpy.data.materials)}):")
    for mat in bpy.data.materials:
        print(f"  - {mat.name}")

    # Locate the 8K skin textures
    basecolor_path = next(tex_dir.glob("*basecolor*.png"), None)
    roughness_path = next(tex_dir.glob("*roughness*.png"), None)
    normal_path = next(tex_dir.glob("*normal*.png"), None)

    if not basecolor_path or not basecolor_path.exists():
        print("Warning: basecolor texture not found directly in textures/, checking alternatives...")
        for p in Path(".").rglob("*skin_basecolor*.png"):
            basecolor_path = p.resolve()
            break

    print(f"BaseColor: {basecolor_path}")
    print(f"Roughness: {roughness_path}")
    print(f"Normal: {normal_path}")

    # Load images into Blender
    img_base = bpy.data.images.load(str(basecolor_path), check_existing=True) if basecolor_path else None
    img_rough = bpy.data.images.load(str(roughness_path), check_existing=True) if roughness_path else None
    img_norm = bpy.data.images.load(str(normal_path), check_existing=True) if normal_path else None

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
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # Check if this object is body/skin or if it's the main character mesh
            name_lower = obj.name.lower()
            # If there's an existing skin or body material, replace it; otherwise add to meshes
            is_body_target = any(k in name_lower for k in ["body", "skin", "head", "face", "torso", "mesh", "character", "ich", "chibi"]) or len(obj.data.materials) == 0
            if is_body_target:
                print(f"Applying Dewy Satin Porcelain skin material to mesh: {obj.name}")
                if len(obj.data.materials) == 0:
                    obj.data.materials.append(skin_mat)
                else:
                    # Find and replace skin/body slots or replace slot 0
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

    # Optional: Render preview
    if args.render_previews:
        preview_path = out_dir / "preview_ICH_skin.png"
        print(f"Rendering preview to: {preview_path}")
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 64
        scene.render.resolution_x = 1000
        scene.render.resolution_y = 1333
        scene.render.filepath = str(preview_path)
        try:
            bpy.ops.render.render(write_still=True)
            print(f"Rendered preview successfully!")
        except Exception as e:
            print(f"Rendering failed: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
