"""Extract ci values from the IN-side heatmap PNG."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path("/home/francisco/Projects/vibeathon")

# Read the heatmap image
img = Image.open(REPO / "spd_decomposition/out/plain_20f_2n/ci_heatmap.png")
arr = np.array(img)
print(f"Image shape: {arr.shape}")   # (H, W, 4) presumably
# We need to find the IN-side subplot. It's the leftmost. We manually find the
# pixel bounds of the heatmap's data area.

# Crop to approximately the IN-side plot area. The figure is 20" x 6" at 150 dpi
# = 3000x900 px. Three subplots. Axis boundaries appear around these pixels.
# Rather than guessing, let's look at the image and find the actual data region.

# Save a cropped version and figure out bounds interactively
print("Image height, width:", arr.shape[0], arr.shape[1])
# The 3 subplots tile widthwise. Approximately each takes ~1/3 of width.

# Let's sample columns to find dark regions (plot frames).
gray = arr[:, :, :3].mean(axis=2)
# Find vertical lines that look like axes (sharp black vertical lines)
col_darkness = (gray < 60).sum(axis=0)
# Find horizontal lines
row_darkness = (gray < 60).sum(axis=1)
print("\nTop 20 darkest columns:")
print(sorted(enumerate(col_darkness), key=lambda x: -x[1])[:20])
print("\nTop 10 darkest rows:")
print(sorted(enumerate(row_darkness), key=lambda x: -x[1])[:10])
