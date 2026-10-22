"""Regenerate the single-metal Pourbaix panels referenced by SI.tex.

Only writes to the local SI preview directory. The source SI uses three
conditions per metal (water-only, NH3/Gly, NH3/Gly/CN), except Mn and Sr.
The first Pd panel follows the referenced H2O image, despite its caption.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_batch_pourbaix import (
    GridPlotter, ROOT, StabilityCalculator, load_complex_dataframe,
    prepare_case,
)
from generate_six_panel_pourbaix import (
    ACTIVITY, DPI, EXPERIMENTAL_PH_RANGE,
    EXPERIMENTAL_POTENTIAL_RANGE, GRID_SIZE, LARGE_PANEL_CM,
    OVERWIDE_LEGEND_CATEGORY_ORDER, PH_RANGE, POTENTIAL_RANGE,
    TEMPERATURE, annotate_au_water_lines,
)


METALS = ("Au", "Pd", "Ni", "Cu", "Fe", "Mn", "Zn", "Cd", "Co", "Sr")
OUTPUT_DIR = ROOT / "examples" / "figures" / "figures" / "si_pourbaix_preview" / "single_metals"
WATER = {"NH3": 0.0, "NO2": 0.0, "Gly": 0.0, "CN": 0.0}
NH3_GLY = {"NH3": 0.02, "NO2": 0.0, "Gly": 0.1, "CN": 0.0}
NH3_GLY_CN = {"NH3": 0.02, "NO2": 0.0, "Gly": 0.1, "CN": 1e-4}
WATER_TITLE = r"$\mathrm{H_2O}$"
NH3_GLY_TITLE = r"$\mathrm{NH_3}$ + Gly"
NH3_GLY_CN_TITLE = r"$\mathrm{NH_3}$ + Gly + $\mathrm{CN^-}$"
CONDITIONS = {
    "Au": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True),
           (NH3_GLY_CN_TITLE, NH3_GLY_CN, True)),
    "Pd": ((WATER_TITLE, WATER, True), (NH3_GLY_CN_TITLE, NH3_GLY_CN, True),
           ("CN: Harrington", NH3_GLY_CN, False)),
    "Ni": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True),
           (NH3_GLY_CN_TITLE, NH3_GLY_CN, True)),
    "Cu": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True),
           (NH3_GLY_CN_TITLE, NH3_GLY_CN, True)),
    "Fe": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True),
           (NH3_GLY_CN_TITLE, NH3_GLY_CN, True)),
    "Mn": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True)),
    "Zn": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True),
           (NH3_GLY_CN_TITLE, NH3_GLY_CN, True)),
    "Cd": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True),
           (NH3_GLY_CN_TITLE, NH3_GLY_CN, True)),
    "Co": ((WATER_TITLE, WATER, True), (NH3_GLY_TITLE, NH3_GLY, True),
           (NH3_GLY_CN_TITLE, NH3_GLY_CN, True)),
    "Sr": (("Gly = 0.1 M", {"NH3": 0.02, "NO2": 0.0,
                            "Gly": 0.1, "CN": 0.0}, True),),
}
HEIGHT_CM = 6.6
STRIDE_CM = 7.65
PLOT_LEFT_CM = 1.0
PLOT_BOTTOM_CM = 1.0
KEY_LEFT_CM = 5.85
KEY_BOTTOM_CM = 0.75
KEY_WIDTH_CM = 2.6
KEY_HEIGHT_CM = 5.2


def parse_args():
    """Read metal, resolution, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metals", nargs="+", choices=METALS, default=list(METALS))
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "pdf", "tif"])
    return parser.parse_args()


def calculate_case(metal, ligands, legacy_override, complex_df, grid_size):
    """Use the same model and grid as the six-panel manuscript figure."""
    data, thermo, species, labels = prepare_case(
        metal, ACTIVITY, TEMPERATURE, ligands, complex_df,
        ligand_correction=False, use_legacy_overrides=legacy_override,
    )
    plotter = GridPlotter(
        pH_range=PH_RANGE, V_range=POTENTIAL_RANGE, data=data,
        grid_size=grid_size, save_fig=False, dir=str(OUTPUT_DIR),
        filename=metal, dpi=DPI,
    )
    regions = StabilityCalculator(
        plotter, species, thermo, data
    ).compute_stable_regions()
    return plotter, regions, labels


def add_side_key(fig, column, width_cm, plotter):
    """Draw a vertically centered, category-sorted colored phase key."""
    key_ax = fig.add_axes((
        (column * STRIDE_CM + KEY_LEFT_CM) / width_cm,
        KEY_BOTTOM_CM / HEIGHT_CM,
        KEY_WIDTH_CM / width_cm,
        KEY_HEIGHT_CM / HEIGHT_CM,
    ))
    key_ax.set_axis_off()
    entries = plotter.color_phase_key
    step = min(0.083, 0.88 / max(len(entries), 1))
    first_y = 0.5 + (len(entries) - 1) * step / 2
    alpha = plotter.style.phase_image_alpha
    for index, (label, color) in enumerate(entries):
        y = first_y - index * step
        blended = tuple(alpha * channel + 1 - alpha for channel in color[:3])
        key_ax.plot(
            0.09, y, marker="o", markersize=5, markerfacecolor=blended,
            markeredgecolor="none", linestyle="none", transform=key_ax.transAxes,
        )
        key_ax.text(
            0.23, y, label, transform=key_ax.transAxes,
            ha="left", va="center", fontsize=7,
        )


def render(metal, cases, output_dir, formats, dpi):
    """Save one Matplotlib panel figure with square phase maps."""
    width_cm = 8.55 + STRIDE_CM * (len(cases) - 1)
    fig = plt.figure(figsize=(width_cm / 2.54, HEIGHT_CM / 2.54))
    for index, (title, plotter, regions, labels) in enumerate(cases):
        left_cm = index * STRIDE_CM + PLOT_LEFT_CM
        ax = fig.add_axes((
            left_cm / width_cm, PLOT_BOTTOM_CM / HEIGHT_CM,
            LARGE_PANEL_CM / width_cm, LARGE_PANEL_CM / HEIGHT_CM,
        ))
        plotter.plot_stable_regions(
            regions, labels, pH_exp_range=EXPERIMENTAL_PH_RANGE,
            V_exp_range=EXPERIMENTAL_POTENTIAL_RANGE,
            label_style="legend", show_legend=False, ax=ax,
            legend_category_order=OVERWIDE_LEGEND_CATEGORY_ORDER,
        )
        if metal == "Au" and index == 0:
            annotate_au_water_lines(ax)
        panel_prefix = f"({chr(97 + index)}) " if len(cases) > 1 else ""
        ax.set_title(f"{panel_prefix}{title}", loc="left", pad=5, fontsize=8)
        ax.set_ylabel("")
        if index:
            ax.tick_params(axis="y", labelleft=False)
        add_side_key(fig, index, width_cm, plotter)
    fig.text(
        0.36 / width_cm, (PLOT_BOTTOM_CM + LARGE_PANEL_CM / 2) / HEIGHT_CM,
        r"$E_{\mathrm{SHE}}$ (V)", rotation=90,
        ha="center", va="center", fontsize=8,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"SI_{metal}_{len(cases)}panel_side_colors"
    for extension in formats:
        output = output_dir / f"{stem}.{extension}"
        fig.savefig(
            output, format="tiff" if extension == "tif" else extension,
            dpi=dpi,
        )
        print(f"saved {output}", flush=True)
    plt.close(fig)


def main():
    """Regenerate SI single-metal figures without changing the source SI."""
    args = parse_args()
    complex_df = load_complex_dataframe()
    for metal in args.metals:
        cases = [
            (title, *calculate_case(
                metal, ligands, legacy_override,
                complex_df, args.grid_size,
            ))
            for title, ligands, legacy_override in CONDITIONS[metal]
        ]
        render(metal, cases, args.output_dir, args.formats, args.dpi)


if __name__ == "__main__":
    main()
