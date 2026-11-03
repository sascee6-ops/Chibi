"""Maximally compressed RGB8 PNG writer (numpy + zlib only, no Blender).

Blender's built-in PNG writer leaves 15-30% on the table compared with a
level-9 zlib encode using per-row filter selection.  At 8192x8192 that
difference decides whether the ``.blend`` with packed texture masters
stays under GitHub's 100 MB per-file git limit, so the pretty pipeline
writes its three skin masters with this encoder instead of
``Image.save()``.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", binascii.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _score(filtered: np.ndarray) -> int:
    """libpng-style heuristic: sum of |signed| filtered byte values."""
    return int(np.abs(filtered.view(np.int8).astype(np.int32)).sum())


def _filter_row(row: np.ndarray, prior: np.ndarray, bpp: int = 3) -> bytes:
    """Return one filtered scanline (filter byte + bytes), best of 5 filters."""
    r = row.astype(np.int16)
    p = prior.astype(np.int16)
    left = np.zeros_like(r)
    left[bpp:] = r[:-bpp]
    upleft = np.zeros_like(p)
    upleft[bpp:] = p[:-bpp]

    paeth_p = left + p - upleft
    pa = np.abs(paeth_p - left)
    pb = np.abs(paeth_p - p)
    pc = np.abs(paeth_p - upleft)
    paeth = np.where(
        (pa <= pb) & (pa <= pc), left, np.where(pb <= pc, p, upleft)
    )

    best_type = 0
    best = row.copy()
    best_score = _score(best)
    for ftype, delta in (
        (1, r - left),
        (2, r - p),
        (3, r - ((left + p) // 2)),
        (4, r - paeth),
    ):
        cand = (delta & 0xFF).astype(np.uint8)
        s = _score(cand)
        if s < best_score:
            best_type, best, best_score = ftype, cand, s
    return bytes([best_type]) + best.tobytes()


def write_png_rgb8(path: Path, rgb: np.ndarray) -> None:
    """Write an HxWx3 uint8 array (top row first) as a level-9 PNG."""
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    height, width, _ = rgb.shape
    flat = rgb.reshape(height, width * 3)

    compressor = zlib.compressobj(9)
    idat = bytearray()
    prior = np.zeros(width * 3, dtype=np.uint8)
    for y in range(height):
        row = flat[y]
        idat += compressor.compress(_filter_row(row, prior))
        prior = row
    idat += compressor.flush()

    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            _chunk(b"IDAT", bytes(idat)),
            _chunk(b"IEND", b""),
        )
    )
    Path(path).write_bytes(png)
