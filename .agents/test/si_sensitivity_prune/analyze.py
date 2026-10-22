"""Compare only the phase-map rectangles of generated sensitivity triptychs."""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("examples/figures/figures/sensitivity_triptychs_with_cn_sweep")
METALS = ("Au", "Ni", "Cu", "Pd", "Pt", "Ti")
PARAMETERS = ("activity", "NH3", "Gly", "CN")
WIDTH_CM = 23.85
HEIGHT_CM = 6.6
STRIDE_CM = 7.65
PLOT_LEFT_CM = 1.0
PLOT_BOTTOM_CM = 1.0
PLOT_SIZE_CM = 4.7


def crop_maps(path: Path) -> list[np.ndarray]:
    """Crop three phase-map interiors, excluding the plot spines."""
    image = np.asarray(Image.open(path).convert("RGB"))
    pixels_per_cm = image.shape[1] / WIDTH_CM
    top = (HEIGHT_CM - PLOT_BOTTOM_CM - PLOT_SIZE_CM) * pixels_per_cm
    bottom = (HEIGHT_CM - PLOT_BOTTOM_CM) * pixels_per_cm
    maps = []
    for panel in range(3):
        left = (PLOT_LEFT_CM + panel * STRIDE_CM) * pixels_per_cm
        right = left + PLOT_SIZE_CM * pixels_per_cm
        margin = 8
        crop = image[
            round(top) + margin : round(bottom) - margin,
            round(left) + margin : round(right) - margin,
        ]
        maps.append(crop)
    return maps


def experimental_window_mask(shape: tuple[int, int]) -> np.ndarray:
    """Return the experimental pH/RHE window in cropped SHE coordinates."""
    pixels_per_cm = 600 / 2.54
    full_side = PLOT_SIZE_CM * pixels_per_cm
    row, column = np.indices(shape)
    ph = -2.0 + 18.0 * (column + 8) / full_side
    e_she = 3.0 - 5.0 * (row + 8) / full_side
    return (
        (ph >= 10.5) & (ph <= 13.5)
        & (e_she >= -2.0 - 0.0591 * ph)
        & (e_she <= 2.3 - 0.0591 * ph)
    )


for metal in METALS:
    for parameter in PARAMETERS:
        path = ROOT / metal / f"{metal}_{parameter}_three_levels_1x3_side_colors.png"
        maps = crop_maps(path)
        mask = experimental_window_mask(maps[0].shape[:2])
        pairwise = []
        window_pairwise = []
        for left, right in ((0, 1), (1, 2), (0, 2)):
            distance = np.max(
                np.abs(maps[left].astype(np.int16) - maps[right].astype(np.int16)),
                axis=2,
            )
            pairwise.append(np.count_nonzero(distance > 16) / distance.size)
            window_pairwise.append(
                np.count_nonzero((distance > 16) & mask) / np.count_nonzero(mask)
            )
        print(
            f"{metal:2} {parameter:8} "
            + " ".join(f"{value:.5f}" for value in pairwise)
            + " | window: "
            + " ".join(f"{value:.5f}" for value in window_pairwise)
        )
