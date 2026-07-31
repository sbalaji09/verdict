from __future__ import annotations

from io import BytesIO

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from verdict.frontend.visual_diff import perceptual_diff_ratio  # noqa: E402


def _solid_png(color: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (100, 100), color).save(buf, format="PNG")
    return buf.getvalue()


def test_identical_images_have_zero_diff() -> None:
    image = _solid_png((20, 20, 20))
    assert perceptual_diff_ratio(image, image) == 0.0


def test_completely_different_images_have_near_total_diff() -> None:
    before = _solid_png((0, 0, 0))
    after = _solid_png((255, 255, 255))
    assert perceptual_diff_ratio(before, after) == pytest.approx(1.0)


def test_small_deltas_are_absorbed_as_render_noise() -> None:
    # A 5/255 grayscale delta is well under PIXEL_TOLERANCE (30) — this is
    # exactly the kind of anti-aliasing/font-rendering jitter the tolerance
    # exists to ignore, so an unrelated page must not flake here.
    before = _solid_png((100, 100, 100))
    after = _solid_png((105, 105, 105))
    assert perceptual_diff_ratio(before, after) == 0.0


def test_large_delta_survives_the_tolerance() -> None:
    before = _solid_png((100, 100, 100))
    after = _solid_png((160, 160, 160))
    assert perceptual_diff_ratio(before, after) == pytest.approx(1.0)
