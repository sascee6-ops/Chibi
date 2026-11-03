"""Re-encode only the base-colour image inside a GLB as high-quality JPEG.

Blender's glTF exporter applies one image format to every embedded map,
and JPEG would corrupt the roughness/normal *data* maps.  At 8192x8192 a
fully lossless GLB (all PNG) exceeds GitHub's 100 MB per-file git limit,
so this post-step swaps just the base-colour PNG for a quality-92 JPEG
at the *same* 8192x8192 resolution.  The PNG masters stay in
``textures/`` and packed losslessly in the ``.blend``.

Usage: python scripts/repack_glb_basecolor.py Chibi_pretty.glb [--quality 92]
"""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def repack(path: Path, quality: int) -> None:
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2 and total == len(data), (magic, version, total)
    json_len, json_type = struct.unpack_from("<I4s", data, 12)
    assert json_type == b"JSON"
    gltf = json.loads(data[20 : 20 + json_len])
    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<I4s", data, bin_off)
    assert bin_type == b"BIN\x00"
    bin_start = bin_off + 8
    old_bin = data[bin_start : bin_start + bin_len]

    tex_index = None
    for mat in gltf.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        if "baseColorTexture" in pbr:
            tex_index = pbr["baseColorTexture"]["index"]
            break
    assert tex_index is not None, "no baseColorTexture found"
    img_index = gltf["textures"][tex_index]["source"]
    image = gltf["images"][img_index]
    view_index = image["bufferView"]
    view = gltf["bufferViews"][view_index]
    start = view.get("byteOffset", 0)
    blob = old_bin[start : start + view["byteLength"]]

    im = Image.open(io.BytesIO(blob))
    print(
        f"base-colour image '{image.get('name')}': {im.size[0]}x{im.size[1]}, "
        f"{image.get('mimeType')}, {len(blob)} bytes"
    )
    if image.get("mimeType") == "image/jpeg":
        print("already JPEG; nothing to do")
        return
    buf = io.BytesIO()
    # 4:2:2 chroma: colour detail of the painted skin survives, size drops hard
    im.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=1)
    new_blob = buf.getvalue()
    print(f"re-encoded JPEG q{quality}: {len(new_blob)} bytes")
    if len(new_blob) >= len(blob):
        print("JPEG is not smaller; keeping the original GLB")
        return

    # rebuild the BIN chunk view by view, swapping in the JPEG bytes
    views = gltf["bufferViews"]
    order = sorted(range(len(views)), key=lambda i: views[i].get("byteOffset", 0))
    new_bin = bytearray()
    for i in order:
        v = views[i]
        s = v.get("byteOffset", 0)
        chunk = new_blob if i == view_index else old_bin[s : s + v["byteLength"]]
        new_bin.extend(b"\x00" * ((-len(new_bin)) % 4))
        v["byteOffset"] = len(new_bin)
        v["byteLength"] = len(chunk)
        new_bin.extend(chunk)

    image["mimeType"] = "image/jpeg"
    gltf["buffers"][0]["byteLength"] = len(new_bin)

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode()
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    bin_bytes = bytes(new_bin) + b"\x00" * ((-len(new_bin)) % 4)

    out = bytearray()
    out += struct.pack(
        "<4sII", b"glTF", 2, 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    )
    out += struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    out += struct.pack("<I4s", len(bin_bytes), b"BIN\x00") + bin_bytes
    path.write_bytes(bytes(out))
    print(f"{path}: {len(data)} -> {len(out)} bytes")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("glb", type=Path)
    p.add_argument("--quality", type=int, default=92)
    args = p.parse_args()
    repack(args.glb, args.quality)


if __name__ == "__main__":
    main()
