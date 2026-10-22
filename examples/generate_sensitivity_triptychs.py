"""Generate one-row, three-condition Pourbaix figures for each metal.

The reference condition matches ``generate_six_panel_pourbaix.py``: ion
activity corresponding to 1e-4 M in the ideal-solution approximation,
total NH3 0.02 M, total Gly 0.1 M, and no CN or NO2. Each figure varies
one concentration at a time. The CN sweep compares 0, 1e-4, and 1e-3 M.
Run from the repository root with::

    python examples/generate_sensitivity_triptychs.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_batch_pourbaix import (
    GridPlotter,
    ROOT,
    StabilityCalculator,
    load_complex_dataframe,
    prepare_case,
)
from generate_six_panel_pourbaix import (
    DPI,
    EXPERIMENTAL_PH_RANGE,
    EXPERIMENTAL_POTENTIAL_RANGE,
    GRID_SIZE,
    LARGE_PANEL_CM,
    METALS,
    OVERWIDE_LEGEND_CATEGORY_ORDER,
    PH_RANGE,
    POTENTIAL_RANGE,
    TEMPERATURE,
    annotate_au_water_lines,
)


BASELINE = {"activity": 1e-4, "NH3": 0.02, "Gly": 0.1, "CN": 0.0}
LEVELS = {
    "activity": (1e-6, 1e-4, 1e-2),
    "NH3": (2e-3, 2e-2, 1e-1),
    "Gly": (1e-2, 1e-1, 1.0),
    "CN": (0.0, 1e-4, 1e-3),
}
PARAMETER_LABELS = {
    "activity": r"$[\mathrm{M}]_{\mathrm{aq}}$",
    "NH3": r"$[\mathrm{NH_3}]_{\mathrm{tot}}$",
    "Gly": r"$[\mathrm{Gly}]_{\mathrm{tot}}$",
    "CN": r"$[\mathrm{CN}]_{\mathrm{tot}}$",
}
OUTPUT_DIR = (
    ROOT / "examples" / "figures" / "figures"
    / "sensitivity_triptychs_with_cn_sweep"
)
FIGURE_WIDTH_CM = 23.85
FIGURE_HEIGHT_CM = 6.6
COLUMN_STRIDE_CM = 7.65
PLOT_LEFT_CM = 1.0
PLOT_BOTTOM_CM = 1.0
KEY_LEFT_CM = 5.85
KEY_BOTTOM_CM = 0.75
KEY_WIDTH_CM = 2.6
KEY_HEIGHT_CM = 5.2


def parse_args():
    """Parse the metal, parameter, quality, and output selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metals", nargs="+", choices=METALS, default=list(METALS))
    parser.add_argument(
        "--parameters", nargs="+", choices=tuple(LEVELS),
        default=list(LEVELS),
    )
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "svg", "pdf", "tif"),
        default=["png", "svg", "pdf", "tif"],
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def calculate_case(metal, parameter, value, complex_df, grid_size):
    """Compute one stability grid with only ``parameter`` varied."""
    condition = dict(BASELINE)
    condition[parameter] = value
    ligands = {
        "NH3": condition["NH3"], "NO2": 0.0,
        "Gly": condition["Gly"], "CN": condition["CN"],
    }
    data, thermo, species, labels = prepare_case(
        metal, condition["activity"], TEMPERATURE, ligands,
        complex_df, ligand_correction=False,
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


def add_side_key(figure, column, plotter):
    """Draw the same color-dot phase key as the overwide six-panel figure."""
    left = column * COLUMN_STRIDE_CM + KEY_LEFT_CM
    key_ax = figure.add_axes((
        left / FIGURE_WIDTH_CM, KEY_BOTTOM_CM / FIGURE_HEIGHT_CM,
        KEY_WIDTH_CM / FIGURE_WIDTH_CM, KEY_HEIGHT_CM / FIGURE_HEIGHT_CM,
    ))
    key_ax.set_axis_off()
    entries = plotter.color_phase_key
    step = min(0.083, 0.88 / max(len(entries), 1))
    first_y = 0.5 + (len(entries) - 1) * step / 2
    alpha = plotter.style.phase_image_alpha
    for index, (label, color) in enumerate(entries):
        y = first_y - index * step
        blended_color = tuple(
            alpha * channel + (1 - alpha) for channel in color[:3]
        )
        key_ax.plot(
            0.09, y, marker="o", markersize=5,
            markerfacecolor=blended_color, markeredgecolor="none",
            linestyle="none", transform=key_ax.transAxes,
        )
        key_ax.text(
            0.23, y, label, transform=key_ax.transAxes,
            ha="left", va="center", fontsize=7,
        )


def render_triptych(metal, parameter, cases, output_dir, formats, dpi):
    """Render three square panels with a centered side key for each panel."""
    figure = plt.figure(figsize=(
        FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54,
    ))
    for column, (value, plotter, regions, labels) in enumerate(cases):
        left = column * COLUMN_STRIDE_CM + PLOT_LEFT_CM
        axis = figure.add_axes((
            left / FIGURE_WIDTH_CM, PLOT_BOTTOM_CM / FIGURE_HEIGHT_CM,
            LARGE_PANEL_CM / FIGURE_WIDTH_CM,
            LARGE_PANEL_CM / FIGURE_HEIGHT_CM,
        ))
        plotter.plot_stable_regions(
            regions, labels,
            pH_exp_range=EXPERIMENTAL_PH_RANGE,
            V_exp_range=EXPERIMENTAL_POTENTIAL_RANGE,
            label_style="legend", show_legend=False, ax=axis,
            legend_category_order=OVERWIDE_LEGEND_CATEGORY_ORDER,
        )
        if metal == "Au" and column == 0:
            annotate_au_water_lines(axis)
        axis.set_title(
            f"({chr(97 + column)}) {PARAMETER_LABELS[parameter]} "
            f"= {value:g} M",
            loc="left", pad=5, fontsize=8,
        )
        axis.set_ylabel("")
        if column:
            axis.tick_params(axis="y", labelleft=False)
        add_side_key(figure, column, plotter)

    figure.text(
        0.015, (PLOT_BOTTOM_CM + LARGE_PANEL_CM / 2) / FIGURE_HEIGHT_CM,
        r"$E_{\mathrm{SHE}}$ (V)", rotation=90,
        ha="center", va="center", fontsize=8,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metal}_{parameter}_three_levels_1x3_side_colors"
    for extension in formats:
        output_path = output_dir / f"{stem}.{extension}"
        figure.savefig(
            output_path, format="tiff" if extension == "tif" else extension,
            dpi=dpi,
        )
        print(f"saved {output_path}", flush=True)
    plt.close(figure)


def main():
    """Calculate and save one concentration-sensitivity triptych per case."""
    args = parse_args()
    complex_df = load_complex_dataframe()
    for metal in args.metals:
        for parameter in args.parameters:
            cases = [
                (value, *calculate_case(
                    metal, parameter, value, complex_df, args.grid_size
                ))
                for value in LEVELS[parameter]
            ]
            render_triptych(
                metal, parameter, cases, args.output_dir / metal,
                args.formats, args.dpi,
            )


if __name__ == "__main__":
    main()
