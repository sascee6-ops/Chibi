from PIL import Image, ImageOps
import numpy as np
from collections import deque

def remove_outer_background(im: Image.Image, bg_thresh=242) -> Image.Image:
    """Flood-fills from the borders to remove ONLY the outer white background,
    preserving all inner specular highlights and whites of the eye."""
    rgba = im.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]

    is_white = (arr[:, :, 0] >= bg_thresh) & (arr[:, :, 1] >= bg_thresh) & (arr[:, :, 2] >= bg_thresh)

    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    for x in range(w):
        if is_white[0, x]:
            queue.append((0, x))
            visited[0, x] = True
        if is_white[h - 1, x]:
            queue.append((h - 1, x))
            visited[h - 1, x] = True
    for y in range(h):
        if is_white[y, 0]:
            queue.append((y, 0))
            visited[y, 0] = True
        if is_white[y, w - 1]:
            queue.append((y, w - 1))
            visited[y, w - 1] = True

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and is_white[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    arr[visited, 3] = 0
    return Image.fromarray(arr)


# Master image
master = Image.open("textures/anime_eye_master.png")

# 1. Front View Open Eye (Character Right / Viewer Left, -X):
# Wing on left, tear duct on right
crop_open_l = master.crop((15, 15, 385, 245))
clean_open_l = remove_outer_background(crop_open_l)
clean_open_l.save("textures/eye_open_L.png")

# 2. Front View Open Eye (Character Left / Viewer Right, +X):
# Exact horizontal mirror for 100% perfect symmetry!
clean_open_r = ImageOps.mirror(clean_open_l)
clean_open_r.save("textures/eye_open_R.png")

# 3. Closed Eye (tight crop excluding text label)
crop_closed = master.crop((890, 380, 1260, 520))
clean_closed = remove_outer_background(crop_closed)
clean_closed.save("textures/eye_closed.png")

print("Flood-fill transparency successfully applied to all eye states with exact mirror symmetry!")
