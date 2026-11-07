"""Comprehensive diagnosis and fix for all black spots / overlapping geometry in ICHxTenebra."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import bpy
import bmesh
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="ICHxTenebra.v0.0.0.0.0.0-s0.blend")
    p.add_argument("--out", default="ICHxTenebra_with_skin.blend")
    p.add_argument("--textures", default="textures")
    p.add_argument("--report", default="preview/ich_diagnostics.txt")
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

    log(f"=== Opening Original Blend File: {blend_path} ===")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    scene = bpy.context.scene
    log(f"Scene name: {scene.name}")
    log(f"View layers ({len(scene.view_layers)}): {[vl.name for vl in scene.view_layers]}")
    log(f"Collections ({len(bpy.data.collections)}): {[c.name for c in bpy.data.collections]}")

    log(f"\n=== Listing ALL {len(bpy.data.objects)} objects in bpy.data.objects ===")
    for o in bpy.data.objects:
        inst_info = f", instance_type={o.instance_type}" if hasattr(o, "instance_type") else ""
        if hasattr(o, "instance_collection") and o.instance_collection:
            inst_info += f", inst_collection='{o.instance_collection.name}'"
        log(f"  Object: '{o.name}' (type={o.type}{inst_info}, hide_render={o.hide_render}, hide_viewport={o.hide_get()})")

    # Step 1: Remove any collection instances / empties duplicating geometry
    removed_instances = 0
    for o in list(bpy.data.objects):
        if o.type == "EMPTY" and hasattr(o, "instance_type") and o.instance_type in ["COLLECTION", "OBJECT"]:
            log(f"REMOVING collection instance Empty: '{o.name}' (instanced {o.instance_collection})")
            bpy.data.objects.remove(o, do_unlink=True)
            removed_instances += 1
    log(f"Removed {removed_instances} collection instances.")

    # Step 2: Inspect character mesh in detail
    char_obj = bpy.data.objects.get("reVAmped_character_female_8k")
    if not char_obj:
        for o in bpy.data.objects:
            if o.type == "MESH" and "cyclorama" not in o.name.lower():
                char_obj = o
                break

    log(f"\nTarget character mesh: '{char_obj.name}'")
    log(f"Modifiers: {[m.name + '(' + m.type + ')' for m in char_obj.modifiers]}")
    log(f"Shape keys: {char_obj.data.shape_keys.key_blocks.keys() if char_obj.data.shape_keys else 'None'}")
    log(f"Vertex groups count: {len(char_obj.vertex_groups)}")
    log(f"UV layers: {[uv.name for uv in char_obj.data.uv_layers]}")
    log(f"Color attributes / vertex colors: {[ca.name for ca in char_obj.data.color_attributes] if hasattr(char_obj.data, 'color_attributes') else 'None'}")

    # Inspect mesh connectivity using bmesh: find loose components
    bm = bmesh.new()
    bm.from_mesh(char_obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Find connected components of faces
    visited_faces = set()
    components = []
    for face in bm.faces:
        if face in visited_faces:
            continue
        comp = []
        queue = [face]
        visited_faces.add(face)
        while queue:
            f = queue.pop()
            comp.append(f)
            for edge in f.edges:
                for linked in edge.link_faces:
                    if linked not in visited_faces:
                        visited_faces.add(linked)
                        queue.append(linked)
        components.append(comp)

    log(f"\nMesh connectivity: found {len(components)} connected polygon islands / shells in '{char_obj.name}':")
    for idx, comp in enumerate(components):
        bbox_min = Vector((min(f.calc_center_median().x for f in comp), min(f.calc_center_median().y for f in comp), min(f.calc_center_median().z for f in comp)))
        bbox_max = Vector((max(f.calc_center_median().x for f in comp), max(f.calc_center_median().y for f in comp), max(f.calc_center_median().z for f in comp)))
        log(f"  Island #{idx+1}: {len(comp)} faces, bbox min={bbox_min}, max={bbox_max}")

    # If there are multiple islands: is one island a duplicate / suit / inner shell?
    # If two islands span the whole body height (e.g. from feet z~0 to head z~77), one is an overlapping duplicate shell!
    if len(components) > 1:
        log("\nAnalyzing multi-island structure for duplicate / overlapping coplanar shells...")
        body_shells = []
        for idx, comp in enumerate(components):
            z_min = min(f.calc_center_median().z for f in comp)
            z_max = max(f.calc_center_median().z for f in comp)
            z_span = z_max - z_min
            log(f"  Island #{idx+1}: z_span={z_span:.2f} (from {z_min:.2f} to {z_max:.2f}), faces={len(comp)}")
            if z_span > 50.0:  # Spans almost full body
                body_shells.append((idx, comp, len(comp)))

        if len(body_shells) > 1:
            log(f"CRITICAL: Found {len(body_shells)} full-body shells inside the same mesh! This is the source of Z-fighting!")
            # Sort by face count descending, keep the primary shell (highest face count / outermost)
            body_shells.sort(key=lambda x: x[2], reverse=True)
            log(f"Keeping primary shell #{body_shells[0][0]+1} with {body_shells[0][2]} faces.")
            # Delete faces of the secondary duplicate shells
            for shell_idx, comp, f_count in body_shells[1:]:
                log(f"Deleting secondary duplicate shell #{shell_idx+1} with {f_count} faces...")
                for f in comp:
                    if f.is_valid:
                        bm.faces.remove(f)

    # Clean unused vertices / edges
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(char_obj.data)
    bm.free()
    char_obj.data.update()

    log(f"\nAfter shell cleanup: {len(char_obj.data.vertices)} vertices, {len(char_obj.data.polygons)} polygons.")

    # Remove doubles / merge coplanar vertices and ensure outward normals
    bpy.context.view_layer.objects.active = char_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Step 3: Check modifiers that could duplicate or create inner/outer shells
    for mod in list(char_obj.modifiers):
        log(f"Inspecting modifier: {mod.name} (type={mod.type})")
        if mod.type in ["SOLIDIFY", "SHRINKWRAP", "BOOLEAN", "MIRROR"]:
            log(f"  REMOVING shell-duplicating modifier: {mod.name}")
            char_obj.modifiers.remove(mod)

    # Step 4: Create Dewy Satin Porcelain 8K skin material
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

    log(f"\nTexture files:")
    log(f"  BaseColor: {basecolor_path}")
    log(f"  Roughness: {roughness_path}")
    log(f"  Normal: {normal_path}")

    img_base = bpy.data.images.load(str(basecolor_path), check_existing=True) if basecolor_path and basecolor_path.exists() else None
    img_rough = bpy.data.images.load(str(roughness_path), check_existing=True) if roughness_path and roughness_path.exists() else None
    img_norm = bpy.data.images.load(str(normal_path), check_existing=True) if normal_path and normal_path.exists() else None

    if img_base:
        img_base.colorspace_settings.name = "sRGB"
    if img_rough:
        img_rough.colorspace_settings.name = "Non-Color"
    if img_norm:
        img_norm.colorspace_settings.name = "Non-Color"

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

    # Clear all materials from character and assign only skin_mat
    char_obj.data.materials.clear()
    char_obj.data.materials.append(skin_mat)
    for p in char_obj.data.polygons:
        p.material_index = 0
    log(f"Assigned DewySatinPorcelain_8K to 100% of character polygons ({len(char_obj.data.polygons)}).")

    # Pack textures
    for img in [img_base, img_rough, img_norm]:
        if img:
            try:
                img.pack()
                log(f"Packed texture: {img.name}")
            except Exception as e:
                log(f"Failed to pack {img.name}: {e}")

    # Save cleaned scene
    log(f"Saving fixed blend file to: {out_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)

    # Write report
    report_file = out_dir / args.report
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"Report written to: {report_file}")

    # Step 5: Render Previews
    if args.render_previews:
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
