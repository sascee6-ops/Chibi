"""Merge the body from ``body base.blend`` with ``Chibi_head.stl``.

Run with Blender, for example:

    blender --background --factory-startup -noaudio \
        --python scripts/merge_chibi.py -- \
        --body "body base.blend" --head Chibi_head.stl \
        --out-dir . --head-scale 0.60

The body blend contains a rigged female body plus a separate neck mesh and
armature bones ``bip_neck``/``bip_head``.  Measured locations (Blender units):

* body mesh:  Z -0.23..59.00, front of the body faces -Y
* neck mesh:  X -2.54..2.54, Y -1.27..3.06, Z 58.58..61.46
* bip_head:   (0, 0.35, 63.03)

The STL head is stored in Blender-like coordinates but its face points to
+Y, the opposite way to the body, so the head is rotated 180 degrees around
Z.  A point on the measured underside of the head is placed on the neck axis
slightly below the neck top, so the neck cylinder disappears into the head
and there is no visible gap.

Outputs (in ``--out-dir``):
    Chibi_full.blend          one joined mesh object named Chibi_full
    Chibi_full.stl            binary STL of the joined mesh
    Chibi_full.glb            glTF binary for easy previewing
    Chibi_full_report.json    geometry report of what was built
    Chibi_full_fused.stl      only with --voxel-size > 0
                              (voxel-remeshed single fused surface)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import bpy  # must be imported before bmesh when running via the bpy wheel
import bmesh
import numpy as np
from mathutils import Matrix, Vector

BODY_OBJECT = "reVAmped_body_base_female_standard"
NECK_OBJECT = "reVAmped_neck_base_female_standard"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--body", default="body base.blend")
    p.add_argument("--head", default="Chibi_head.stl")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--name", default="Chibi_full")
    p.add_argument("--head-scale", type=float, default=0.60)
    p.add_argument(
        "--neck-penetration",
        type=float,
        default=1.16,
        help="how far below the neck top the head underside sits",
    )
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.30,
        help="if > 0, also write a voxel-remeshed fused STL with this voxel size",
    )
    args = p.parse_args(argv)
    if args.head_scale <= 0:
        p.error("--head-scale must be positive")
    if args.voxel_size < 0:
        p.error("--voxel-size must not be negative")
    return args


def vertices_np(mesh: bpy.types.Mesh) -> np.ndarray:
    arr = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", arr)
    return arr.reshape(-1, 3)


def set_vertices_np(mesh: bpy.types.Mesh, verts: np.ndarray) -> None:
    mesh.vertices.foreach_set("co", np.asarray(verts, dtype=np.float64).ravel())
    mesh.update()


def world_bbox(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    arr = np.array([tuple(c) for c in corners], dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def signed_volume(mesh: bpy.types.Mesh) -> float:
    mesh.calc_loop_triangles()
    verts = vertices_np(mesh)
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    tv = verts[tris]
    return float(np.einsum("ij,ij->", tv[:, 0], np.cross(tv[:, 1], tv[:, 2])) / 6.0)


def ensure_outward_normals(obj: bpy.types.Object) -> None:
    """Make face normals consistent; flip a closed mesh if it is inside-out."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    if signed_volume(mesh) < 0:  # closed solid wound inwards
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()


def select_only(objs, active=None) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    if active is not None:
        bpy.context.view_layer.objects.active = active


# --------------------------------------------------------------------------- #
# STL loading (raw binary coordinates, no importer axis conversion surprises)
# --------------------------------------------------------------------------- #
def load_binary_stl(path: Path, name: str) -> bpy.types.Object:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("STL file is too short")
    triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + triangles * 50
    if len(data) < expected:
        raise ValueError(
            f"Binary STL is truncated: {len(data)} bytes, expected {expected}"
        )

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    index: dict[tuple[float, float, float], int] = {}
    offset = 84
    for _ in range(triangles):
        vals = struct.unpack_from("<12fH", data, offset)
        offset += 50
        face = []
        for i in (3, 6, 9):
            vert = (float(vals[i]), float(vals[i + 1]), float(vals[i + 2]))
            found = index.get(vert)
            if found is None:
                found = len(verts)
                index[vert] = found
                verts.append(vert)
            face.append(found)
        if len(set(face)) == 3:
            faces.append(tuple(face))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------------- #
# body baking
# --------------------------------------------------------------------------- #
def bake_object(src: bpy.types.Object, depsgraph, new_name: str) -> bpy.types.Object:
    """Create a world-space copy with modifiers/pose applied."""
    evaluated = src.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    obj = bpy.data.objects.new(new_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.transform(evaluated.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    mesh.update()
    return obj


def remove_all_except(keep: set[str]) -> None:
    for obj in list(bpy.data.objects):
        if obj.name not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)


def purge_orphans() -> None:
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        except Exception:
            break


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_stl(obj: bpy.types.Object, filepath: Path) -> None:
    select_only([obj], active=obj)
    if hasattr(bpy.ops.wm, "stl_export"):  # Blender >= 4.0
        bpy.ops.wm.stl_export(
            filepath=str(filepath),
            export_selected_objects=True,
            use_scene_unit=False,
            global_scale=1.0,
            forward_axis="Y",
            up_axis="Z",
            apply_modifiers=True,
        )
    else:  # Blender < 4.0 legacy add-on
        bpy.ops.export_mesh.stl(
            filepath=str(filepath),
            use_selection=True,
            use_mesh_modifiers=True,
            axis_forward="Y",
            axis_up="Z",
        )


def export_glb(obj: bpy.types.Object, filepath: Path) -> None:
    select_only([obj], active=obj)
    # Default image handling embeds the baked maps losslessly (PNG) at
    # full resolution inside the GLB; do not force JPEG here, it would
    # also recompress the roughness/normal data maps.
    bpy.ops.export_scene.gltf(
        filepath=str(filepath),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,  # glTF standard: Y-up
        export_animations=False,
    )


def mesh_report(obj: bpy.types.Object) -> dict:
    mesh = obj.data
    mesh.calc_loop_triangles()
    mn, mx = world_bbox(obj)
    return {
        "name": obj.name,
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.polygons),
        "triangles": len(mesh.loop_triangles),
        "bbox_min": [round(float(v), 4) for v in mn],
        "bbox_max": [round(float(v), 4) for v in mx],
        "dimensions": [round(float(mx[i] - mn[i]), 4) for i in range(3)],
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    body_path = Path(args.body).resolve()
    head_path = Path(args.head).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not body_path.is_file():
        raise FileNotFoundError(body_path)
    if not head_path.is_file():
        raise FileNotFoundError(head_path)

    print(f"Blender {bpy.app.version_string}")
    print(f"Opening body: {body_path}")
    bpy.ops.wm.open_mainfile(filepath=str(body_path))
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for required in (BODY_OBJECT, NECK_OBJECT):
        if required not in bpy.data.objects:
            raise RuntimeError(f"Object {required!r} not found in body blend")

    body = bake_object(bpy.data.objects[BODY_OBJECT], depsgraph, "Body_baked")
    neck = bake_object(bpy.data.objects[NECK_OBJECT], depsgraph, "Neck_baked")
    body_stats = mesh_report(body)
    neck_stats = mesh_report(neck)
    neck_min, neck_max = world_bbox(neck)
    neck_center = (neck_min + neck_max) / 2.0
    print(f"Body baked: {body_stats}")
    print(f"Neck baked: {neck_stats}")

    remove_all_except({body.name, neck.name})

    # --- head -------------------------------------------------------------
    head = load_binary_stl(head_path, "Chibi_head")
    raw_verts = vertices_np(head.data)
    raw_min, raw_max = raw_verts.min(axis=0), raw_verts.max(axis=0)
    print(f"Head raw STL bbox min={raw_min} max={raw_max}")
    if not (20.0 < raw_max[2] - raw_min[2] < 40.0):
        raise RuntimeError("Unexpected head dimensions; refusing to guess placement")

    # Point on the measured head underside (sagittal centre, roughly under the
    # skull centre).  It is placed onto the neck axis.
    socket_raw = np.array(
        [
            (raw_min[0] + raw_max[0]) / 2.0,
            raw_min[1] + 0.527 * (raw_max[1] - raw_min[1]),
            raw_min[2] + 0.57,
        ]
    )
    target = np.array(
        [neck_center[0], neck_center[1], neck_max[2] - args.neck_penetration]
    )
    # Face of the STL points to +Y; the body front is -Y -> rotate 180 deg in Z.
    rot = np.diag([-1.0, -1.0, 1.0]) * float(args.head_scale)
    placed = target + (raw_verts - socket_raw) @ rot.T
    set_vertices_np(head.data, placed)
    ensure_outward_normals(head)
    bpy.context.view_layer.update()
    head_stats = mesh_report(head)
    print(f"Head placed (scale={args.head_scale}): {head_stats}")

    # --- join into a single mesh object -----------------------------------
    select_only([body, neck, head], active=body)
    bpy.ops.object.join()
    full = bpy.context.view_layer.objects.active
    full.name = args.name
    full.data.name = args.name
    full.matrix_world = Matrix.Identity(4)
    full.vertex_groups.clear()
    if full.data.shape_keys:
        full.shape_key_clear()
    for modifier in list(full.modifiers):
        full.modifiers.remove(modifier)

    # One neutral material for the whole figure (STL keeps no materials).
    mat = bpy.data.materials.get("Chibi_skin") or bpy.data.materials.new("Chibi_skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.93, 0.76, 0.66, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.65
    full.data.materials.clear()
    full.data.materials.append(mat)

    remove_all_except({full.name})
    purge_orphans()
    bpy.context.view_layer.update()
    full_stats = mesh_report(full)
    print(f"Joined: {full_stats}")

    outputs: dict[str, str] = {}

    blend_path = out_dir / f"{args.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
    outputs["blend"] = blend_path.name

    stl_path = out_dir / f"{args.name}.stl"
    export_stl(full, stl_path)
    outputs["stl"] = stl_path.name

    glb_path = out_dir / f"{args.name}.glb"
    export_glb(full, glb_path)
    outputs["glb"] = glb_path.name

    # --- optional: a really fused single surface --------------------------
    fused_stats = None
    if args.voxel_size and args.voxel_size > 0:
        fused = full.copy()
        fused.data = full.data.copy()
        fused.name = f"{args.name}_fused"
        bpy.context.scene.collection.objects.link(fused)
        fused.data.remesh_voxel_size = float(args.voxel_size)
        fused.data.remesh_voxel_adaptivity = 0.0
        select_only([fused], active=fused)
        bpy.ops.object.voxel_remesh()
        fused_stats = mesh_report(fused)
        print(f"Fused (voxel {args.voxel_size}): {fused_stats}")
        fused_path = out_dir / f"{args.name}_fused.stl"
        export_stl(fused, fused_path)
        outputs["fused_stl"] = fused_path.name
        bpy.data.objects.remove(fused, do_unlink=True)

    report = {
        "blender_version": bpy.app.version_string,
        "inputs": {"body": body_path.name, "head": head_path.name},
        "parameters": {
            "head_scale": args.head_scale,
            "head_rotation_z_degrees": 180.0,
            "neck_penetration": args.neck_penetration,
            "head_socket_raw": [round(float(v), 4) for v in socket_raw],
            "head_socket_target": [round(float(v), 4) for v in target],
            "voxel_size": args.voxel_size,
        },
        "body": body_stats,
        "neck": neck_stats,
        "head_placed": head_stats,
        "joined": full_stats,
        "fused": fused_stats,
        "outputs": outputs,
    }
    (out_dir / f"{args.name}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("REPORT " + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
