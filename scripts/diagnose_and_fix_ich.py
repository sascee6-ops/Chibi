"""Diagnose and fix overlapping / black z-fighting geometry in ICHxTenebra scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="ICHxTenebra.v0.0.0.0.0.0-s0.blend")
    p.add_argument("--out", default="ICHxTenebra_fixed.blend")
    p.add_argument("--textures", default="textures")
    p.add_argument("--report", default="ich_diagnostics.txt")
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
    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    log(f"=== Diagnosing Blend File: {blend_path} ===")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    log(f"Blender version: {bpy.app.version_string}")
    log(f"Total objects: {len(bpy.data.objects)}")
    log(f"Total materials: {len(bpy.data.materials)}")

    mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]
    log(f"Mesh objects count: {len(mesh_objects)}")

    for i, obj in enumerate(mesh_objects):
        log(f"\n--- Mesh Object #{i+1}: '{obj.name}' ---")
        log(f"  Visible in viewport: {not obj.hide_get()}, Hide render: {obj.hide_render}")
        log(f"  Vertices: {len(obj.data.vertices)}, Polygons: {len(obj.data.polygons)}, Edges: {len(obj.data.edges)}")
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        center = sum(bbox, Vector()) / 8.0
        dims = (
            max(v.x for v in bbox) - min(v.x for v in bbox),
            max(v.y for v in bbox) - min(v.y for v in bbox),
            max(v.z for v in bbox) - min(v.z for v in bbox),
        )
        log(f"  World center: ({center.x:.3f}, {center.y:.3f}, {center.z:.3f})")
        log(f"  Dimensions: ({dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f})")
        log(f"  Modifiers ({len(obj.modifiers)}): {[m.name + ':' + m.type for m in obj.modifiers]}")
        log(f"  Material slots ({len(obj.material_slots)}):")
        for s_idx, slot in enumerate(obj.material_slots):
            m_name = slot.material.name if slot.material else "<None>"
            # Count polygons using this slot
            poly_count = sum(1 for p in obj.data.polygons if p.material_index == s_idx)
            log(f"    Slot {s_idx}: '{m_name}' -> {poly_count} polygons")
        log(f"  UV layers: {[uv.name for uv in obj.data.uv_layers]}")

    # Inspect all materials in detail
    log("\n=== Materials Detail ===")
    for mat in bpy.data.materials:
        log(f"Material: '{mat.name}', use_nodes={mat.use_nodes}")
        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    base_col = node.inputs["Base Color"].default_value[:] if "Base Color" in node.inputs else "?"
                    rough = node.inputs["Roughness"].default_value if "Roughness" in node.inputs else "?"
                    log(f"  PrincipledBSDF: BaseColor={base_col}, Roughness={rough}")
                elif node.type == "TEX_IMAGE" and node.image:
                    log(f"  ImageTexture: '{node.image.name}' (filepath={node.image.filepath})")

    # Locate skin textures
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

    log(f"\nTextures found:")
    log(f"  BaseColor: {basecolor_path}")
    log(f"  Roughness: {roughness_path}")
    log(f"  Normal: {normal_path}")

    # Load textures
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

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    set_input(bsdf, "Metallic", 0.0)
    set_input(bsdf, "Roughness", 0.46)
    set_input(bsdf, "Subsurface Weight", 0.46)
    set_input(bsdf, "Subsurface Radius", (2.2, 1.1, 0.65))
    set_input(bsdf, "Subsurface Scale", 0.05)
    set_input(bsdf, "Sheen Weight", 0.28)
    set_input(bsdf, "Sheen Tint", (1.0, 0.88, 0.85, 1.0))
    set_input(bsdf, "Coat Weight", 0.12)
    set_input(bsdf, "Coat Roughness", 0.08)

    if img_base:
        tex_base = nodes.new("ShaderNodeTexImage")
        tex_base.image = img_base
        tex_base.location = (-400, 200)
        links.new(tex_base.outputs["Color"], bsdf.inputs["Base Color"])
        if "Subsurface Color" in bsdf.inputs:
            links.new(tex_base.outputs["Color"], bsdf.inputs["Subsurface Color"])

    if img_rough:
        tex_rough = nodes.new("ShaderNodeTexImage")
        tex_rough.image = img_rough
        tex_rough.location = (-400, -50)
        links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])

    if img_norm:
        tex_norm = nodes.new("ShaderNodeTexImage")
        tex_norm.image = img_norm
        tex_norm.location = (-600, -300)
        norm_map = nodes.new("ShaderNodeNormalMap")
        norm_map.location = (-300, -300)
        set_input(norm_map, "Strength", 0.58)
        links.new(tex_norm.outputs["Color"], norm_map.inputs["Color"])
        links.new(norm_map.outputs["Normal"], bsdf.inputs["Normal"])

    # === FIXING GEOMETRY / OVERLAPS / BLACK SPOTS ===
    log("\n=== Analyzing and Resolving Overlapping / Black Areas ===")

    # Case A: Check if there are multiple mesh objects that overlap (coplanar duplicates)
    # Check pairwise bounding boxes and vertex counts
    to_remove = []
    to_hide = []
    for i in range(len(mesh_objects)):
        for j in range(i + 1, len(mesh_objects)):
            o1, o2 = mesh_objects[i], mesh_objects[j]
            bb1 = [o1.matrix_world @ Vector(c) for c in o1.bound_box]
            bb2 = [o2.matrix_world @ Vector(c) for c in o2.bound_box]
            c1 = sum(bb1, Vector()) / 8.0
            c2 = sum(bb2, Vector()) / 8.0
            dist = (c1 - c2).length
            v_ratio = min(len(o1.data.vertices), len(o2.data.vertices)) / max(len(o1.data.vertices), len(o2.data.vertices), 1)
            log(f"Comparison '{o1.name}' vs '{o2.name}': center distance={dist:.4f}, vertex ratio={v_ratio:.4f}")

            # If objects are overlapping in the same position
            if dist < 1.0 and v_ratio > 0.8:
                log(f"ALERT: Detected duplicate/overlapping coplanar mesh pair: '{o1.name}' and '{o2.name}'!")
                # Determine which one is Tenebra / black suit vs ICH / body
                if "tenebra" in o2.name.lower() or "suit" in o2.name.lower() or "black" in o2.name.lower() or "cloth" in o2.name.lower():
                    to_remove.append(o2)
                elif "tenebra" in o1.name.lower() or "suit" in o1.name.lower() or "black" in o1.name.lower() or "cloth" in o1.name.lower():
                    to_remove.append(o1)
                else:
                    # Look at materials
                    mats1 = [s.material.name.lower() for s in o1.material_slots if s.material]
                    mats2 = [s.material.name.lower() for s in o2.material_slots if s.material]
                    log(f"  Materials on '{o1.name}': {mats1}")
                    log(f"  Materials on '{o2.name}': {mats2}")
                    if any("tenebra" in m or "black" in m or "suit" in m or "dark" in m for m in mats2):
                        to_remove.append(o2)
                    elif any("tenebra" in m or "black" in m or "suit" in m or "dark" in m for m in mats1):
                        to_remove.append(o1)
                    else:
                        # Keep the one with higher vertex count or keep o1
                        to_remove.append(o2)

    for o in to_remove:
        log(f"Removing duplicate/coplanar object: '{o.name}'")
        bpy.data.objects.remove(o, do_unlink=True)

    # Re-evaluate remaining mesh objects
    remaining_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    log(f"\nRemaining mesh objects ({len(remaining_meshes)}): {[o.name for o in remaining_meshes]}")

    # Case B: Check within each mesh object for:
    # 1. Multiple material slots with black materials assigned to certain faces
    # 2. Overlapping polygon shells or duplicate vertices inside the same object
    for obj in remaining_meshes:
        log(f"\nProcessing mesh object '{obj.name}':")

        # Disable/remove boolean or shrinkwrap modifiers that cause artifacts if present
        for mod in list(obj.modifiers):
            if mod.type in ["BOOLEAN", "SHRINKWRAP", "LATTICE"]:
                log(f"  Removing problematic modifier: {mod.name} ({mod.type})")
                obj.modifiers.remove(mod)

        # In edit mode: merge duplicate vertices / remove doubles, recalculate normals
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        # Recalculate normals to outside
        bpy.ops.mesh.normals_make_consistent(inside=False)
        # Merge by distance (tiny threshold to clean coplanar duplicates)
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Check material slots: replace ALL material slots with skin_mat (or ensure entire body has skin_mat)
        log(f"  Current material slots: {[s.material.name if s.material else '<empty>' for s in obj.material_slots]}")
        obj.data.materials.clear()
        obj.data.materials.append(skin_mat)
        for poly in obj.data.polygons:
            poly.material_index = 0
        log(f"  Cleaned: object '{obj.name}' now exclusively uses 'DewySatinPorcelain_8K' on all {len(obj.data.polygons)} polygons.")

    # Pack textures
    for img in [img_base, img_rough, img_norm]:
        if img:
            try:
                img.pack()
                log(f"Packed texture: {img.name}")
            except Exception as e:
                log(f"Could not pack {img.name}: {e}")

    # Save cleaned scene
    log(f"Saving fixed blend file to: {out_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)

    # Write report file
    report_file = out_dir / args.report
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"Diagnostics written to {report_file}")

    # Render clean previews
    if args.render_previews:
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        scene.render.resolution_x = 1000
        scene.render.resolution_y = 1333

        # Render original camera
        if scene.camera is not None:
            preview_orig = preview_dir / "ICHxTenebra_preview.png"
            scene.render.filepath = str(preview_orig)
            log(f"Rendering fixed scene to {preview_orig}...")
            try:
                bpy.ops.render.render(write_still=True)
                log("Rendered active camera successfully!")
            except Exception as e:
                log(f"Active camera render failed: {e}")

        # Render studio closeup and front
        active_mesh = remaining_meshes[0] if remaining_meshes else None
        if active_mesh:
            bbox = [active_mesh.matrix_world @ Vector(c) for c in active_mesh.bound_box]
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

    log("Diagnostics and repair finished!")


if __name__ == "__main__":
    main()
