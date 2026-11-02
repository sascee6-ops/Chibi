"""Validate a binary STL and print a small geometry report.

Example:
    python scripts/validate_stl.py Chibi_full_fused.stl \
        --require-watertight --require-single-component \
        --require-positive-volume

Vertices are welded by their exact float32 coordinates (the normal case for
an STL exported by Blender).  A watertight mesh is reported when every edge
is shared by exactly two triangles.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_binary_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("STL file is too short")
    count = struct.unpack_from("<I", data, 80)[0]
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    expected = 84 + count * dtype.itemsize
    if len(data) < expected:
        raise ValueError(f"Binary STL is truncated: {len(data)} of {expected} bytes")
    records = np.frombuffer(data, dtype=dtype, count=count, offset=84)
    return np.asarray(records["verts"], dtype=np.float64)


def weld(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = triangles.reshape(-1, 3)
    index: dict[tuple[float, float, float], int] = {}
    unique: list[tuple[float, float, float]] = []
    faces = np.empty(len(triangles) * 3, dtype=np.int64)
    for i, vertex in enumerate(map(tuple, flat)):
        found = index.get(vertex)
        if found is None:
            found = len(unique)
            index[vertex] = found
            unique.append(vertex)
        faces[i] = found
    return np.asarray(unique, dtype=np.float64), faces.reshape(-1, 3)


def analyze(path: Path) -> dict:
    triangles = load_binary_stl(path)
    vertices, faces = weld(triangles)

    edge_face: dict[tuple[int, int], int] = {}
    edge_counts: dict[tuple[int, int], int] = {}
    components = DisjointSet(len(faces))
    for face_index, face in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[a]), int(face[b]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            if edge in edge_face:
                components.union(face_index, edge_face[edge])
            else:
                edge_face[edge] = face_index

    counts = np.asarray(list(edge_counts.values()), dtype=np.int64)
    component_count = len({components.find(i) for i in range(len(faces))})
    tri_verts = vertices[faces]
    signed_volume = float(
        np.einsum("ij,ij->", tri_verts[:, 0], np.cross(tri_verts[:, 1], tri_verts[:, 2]))
        / 6.0
    )
    normals = np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0])
    area = float(np.linalg.norm(normals, axis=1).sum() / 2.0)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "triangles": int(len(faces)),
        "unique_vertices": int(len(vertices)),
        "unique_edges": int(len(edge_counts)),
        "boundary_edges": int((counts == 1).sum()),
        "edges_used_more_than_twice": int((counts > 2).sum()),
        "max_edge_use": int(counts.max()) if len(counts) else 0,
        "components": component_count,
        "watertight": bool(len(counts) and np.all(counts == 2)),
        "signed_volume": round(signed_volume, 4),
        "surface_area": round(area, 4),
        "bbox_min": [round(float(v), 4) for v in bbox_min],
        "bbox_max": [round(float(v), 4) for v in bbox_max],
        "dimensions": [round(float(bbox_max[i] - bbox_min[i]), 4) for i in range(3)],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stl")
    parser.add_argument("--json-out")
    parser.add_argument("--require-watertight", action="store_true")
    parser.add_argument("--require-single-component", action="store_true")
    parser.add_argument("--require-positive-volume", action="store_true")
    parser.add_argument("--min-triangles", type=int, default=1)
    args = parser.parse_args(argv)

    report = analyze(Path(args.stl))
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    failures = []
    if report["triangles"] < args.min_triangles:
        failures.append(
            f"triangles {report['triangles']} < required {args.min_triangles}"
        )
    if args.require_watertight and not report["watertight"]:
        failures.append("mesh is not watertight")
    if args.require_single_component and report["components"] != 1:
        failures.append(f"mesh has {report['components']} components, expected 1")
    if args.require_positive_volume and report["signed_volume"] <= 0:
        failures.append("signed volume is not positive")
    if failures:
        print("VALIDATION FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
