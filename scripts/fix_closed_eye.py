from PIL import Image
import numpy as np
from collections import deque

master = Image.open("textures/anime_eye_master.png")

# Let's crop closed eye with clean padding
# In master, closed eye is at x: [890, 1260], y: [385, 520]
crop_closed = master.crop((890, 395, 1260, 520))
arr = np.array(crop_closed.convert("RGBA"))
h, w = arr.shape[:2]

# Outer background removal with flood fill, threshold 248
is_white = (arr[:, :, 0] >= 248) & (arr[:, :, 1] >= 248) & (arr[:, :, 2] >= 248)
visited = np.zeros((h, w), dtype=bool)
queue = deque()

# Only seed from bottom and side edges (NOT the top edge where eyelid flesh touches)
for x in range(w):
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
Image.fromarray(arr).save("textures/eye_closed.png")
print("Saved perfect textures/eye_closed.png")
