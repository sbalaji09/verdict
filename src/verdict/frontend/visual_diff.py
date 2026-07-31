"""Perceptual, thresholded before/after screenshot diffing.

Deliberately *not* a raw pixel-exact comparison: the same page rendered
twice by the same browser can differ at the raw-pixel level from font
hinting, anti-aliasing, and sub-pixel layout jitter alone — a raw diff would
flake on a page nobody touched. Two normalizations make this perceptual
instead of literal:

1. Both screenshots are downscaled to a fixed, small size before comparing.
   This is exactly what a human "does this look different" glance does —
   it washes out single-pixel-level noise while still catching a moved
   button, a missing element, or a color change.
2. Per-pixel differences below `PIXEL_TOLERANCE` (out of 255, on a grayscale
   conversion) don't count as "different" at all — only differences bright
   enough to be a real content change survive into the ratio.

The *ratio* of surviving differing pixels to total pixels is what gets
compared against `verdict.yml`'s `frontend.screenshot_threshold` — a
fraction, not a raw count, so the same config value means the same thing at
1440px and at 375px.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageChops

DIFF_DIMENSIONS = (400, 300)
PIXEL_TOLERANCE = 30
"""Grayscale delta (0-255) below which a pixel doesn't count as changed —
absorbs anti-aliasing/font-rendering noise between two renders of the same
unchanged page."""


def _normalize(png_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(png_bytes)).convert("L").resize(DIFF_DIMENSIONS)


def perceptual_diff_ratio(before_png: bytes, after_png: bytes) -> float:
    """Fraction (0.0-1.0) of the normalized image that changed beyond
    `PIXEL_TOLERANCE`."""
    before = _normalize(before_png)
    after = _normalize(after_png)
    diff = ImageChops.difference(before, after)
    thresholded = diff.point(lambda p: 255 if p > PIXEL_TOLERANCE else 0)
    changed_pixels = thresholded.histogram()[-1]
    total_pixels = DIFF_DIMENSIONS[0] * DIFF_DIMENSIONS[1]
    return changed_pixels / total_pixels
