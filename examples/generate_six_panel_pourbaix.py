"""Make a 2-by-3, double-column Pourbaix figure at its final print size.

Run ``python examples/generate_six_panel_pourbaix.py`` from the repository root.
The experimental pH box is 10.5-13.5; the full plot remains pH -2–16.
"""

import argparse
from math import atan2, degrees
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from generate_batch_pourbaix import (
    GridPlotter,
    LIGAND_PRESETS,
    ROOT,
    StabilityCalculator,
    load_complex_dataframe,
    prepare_case,
)


METALS = ("Au", "Ni", "Cu", "Pd", "Pt", "Ti")
ACTIVITY = 1e-4
TEMPERATURE = 298.15
LIGAND_CONCENTRATIONS = LIGAND_PRESETS["nh3_gly_0p1"]
GRID_SIZE = 1000
PH_RANGE = (-2.0, 16.0)
POTENTIAL_RANGE = (-2.0, 3.0)
EXPERIMENTAL_PH_RANGE = (10.5, 13.5)
EXPERIMENTAL_POTENTIAL_RANGE = (-2.0, 2.0)
DPI = 600
FIGURE_WIDTH_CM = 17.1
FIGURE_HEIGHT_CM = 13.2
NUMBERED_FIGURE_HEIGHT_CM = 22.5
SIDE_KEY_FIGURE_HEIGHT_CM = 13.6
THREE_BY_TWO_FIGURE_WIDTH_CM = 16.1
THREE_BY_TWO_FIGURE_HEIGHT_CM = 17.0
TWO_BY_THREE_BELOW_FIGURE_HEIGHT_CM = 18.0
TWO_BY_THREE_OVERWIDE_FIGURE_WIDTH_CM = 23.85
TWO_BY_THREE_OVERWIDE_FIGURE_HEIGHT_CM = 12.0
TWO_BY_THREE_FULL_AXES_WIDTH_CM = 25.85
TWO_BY_THREE_FULL_AXES_HEIGHT_CM = 13.2
LARGE_PANEL_CM = 4.7
OUTPUT_DIR = ROOT / "examples" / "figures" / "figures" / "six_panel_nh3_gly_0p1_no_cn"
OUTPUT_STEM = "Au_Ni_Cu_Pd_Pt_Ti_activity_1e-4_NH3_0p02_Gly_0p1"
OVERWIDE_LEGEND_CATEGORY_ORDER = {
    "metal_oxide": 0,
    "metal_oxyhydroxide": 0,
    "metal_ligand_complex": 1,
    "aqueous_metal_ion": 2,
    "metal": 3,
    "metal_hydride": 4,
}


def add_phase_family_key(fig):
    """Explain the color families without implying that colors identify species."""
    families = (
        ("Oxide / oxyhydroxide", "YlOrBr", 0.65),
        ("Metal / hydride", "Greys", 0.55),
        ("Aqueous ion", "PuBuGn", 0.55),
        ("Ligand complex", "RdPu", 0.55),
    )
    handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markersize=4.5,
            markerfacecolor=plt.get_cmap(colormap)(intensity),
            markeredgecolor="none", label=label,
        )
        for label, colormap, intensity in families
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=4, frameon=False, fontsize=6.5, handletextpad=0.25,
        columnspacing=0.8,
    )


def annotate_au_water_lines(ax):
    """Label Au's water limits parallel to their rendered dashed lines."""
    annotations = (
        ("Oxygen Line", r"$\mathrm{H_2O}$ Oxidation", 0.12, "bottom"),
        ("Hydrogen Line", r"$\mathrm{H_2O}$ Reduction", -0.12, "top"),
    )
    x = 0.0
    for line_label, text_label, offset, vertical_alignment in annotations:
        line = next(line for line in ax.lines if line.get_label() == line_label)
        x_values, y_values = line.get_data()
        slope = (y_values[-1] - y_values[0]) / (x_values[-1] - x_values[0])
        y = y_values[0] + slope * (x - x_values[0])
        start = ax.transData.transform((x, y))
        end = ax.transData.transform((x + 1, y + slope))
        angle = degrees(atan2(end[1] - start[1], end[0] - start[0]))
        ax.text(
            x, y + offset, text_label,
            color=line.get_color(), fontsize=7, rotation=angle,
            rotation_mode="anchor", ha="left", va=vertical_alignment,
            zorder=5,
        )


def main(only_overwide_side_colors=False):
    """Calculate six cases and render all or just the overwide side-color figure."""
    complex_df = load_complex_dataframe()
    cases = []
    if not only_overwide_side_colors:
        fig, axes = plt.subplots(
            2, 3, sharex=True, sharey=True,
            figsize=(FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54),
        )
        fig.subplots_adjust(
            left=0.10, right=0.985, bottom=0.19, top=0.94,
            wspace=0.13, hspace=0.20,
            # wspace=0.05, hspace=0.05,
        )

    for index, metal in enumerate(METALS):
        row, column = divmod(index, 3)
        data, thermo, species, labels = prepare_case(
            metal, ACTIVITY, TEMPERATURE, LIGAND_CONCENTRATIONS,
            complex_df, ligand_correction=False,
        )
        plotter = GridPlotter(
            pH_range=PH_RANGE,
            V_range=POTENTIAL_RANGE,
            data=data,
            grid_size=GRID_SIZE,
            save_fig=False,
            dir=str(OUTPUT_DIR),
            filename=OUTPUT_STEM,
            dpi=DPI,
        )
        stable_regions = StabilityCalculator(
            plotter, species, thermo, data
        ).compute_stable_regions()
        cases.append((metal, plotter, stable_regions, labels))
        if only_overwide_side_colors:
            continue
        ax = axes[row, column]
        plotter.plot_stable_regions(
            stable_regions, labels,
            avoid_label_overlap=True,
            label_indicator="line",
            pH_exp_range=EXPERIMENTAL_PH_RANGE,
            V_exp_range=EXPERIMENTAL_POTENTIAL_RANGE,
            label_style="direct",
            show_legend=False,
            ax=ax,
        )
        ax.set_title(f"({chr(97 + index)}) {metal}", loc="left", pad=2, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)
        if column > 0:
            ax.tick_params(axis="y", labelleft=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if only_overwide_side_colors:
        render_side_legends(cases, overwide=True)
        return

    fig.supxlabel("pH", y=0.105, fontsize=8)
    fig.supylabel(r"$E_{\mathrm{SHE}}$ (V)", x=0.015, fontsize=8)
    add_phase_family_key(fig)

    for extension in ("png", "svg", "pdf", "tif"):
        output_path = OUTPUT_DIR / f"{OUTPUT_STEM}.{extension}"
        fig.savefig(
            output_path,
            format="tiff" if extension == "tif" else extension,
            dpi=DPI,
        )
        print(f"saved {output_path}")
    plt.close(fig)
    # render_numbered(cases)
    render_side_legends(cases)
    render_side_legends(cases, columns=2)
    render_two_by_three_below_legends(cases)
    render_side_legends(cases, overwide=True)
    render_side_legends(cases, overwide=True, axes_mode="upper_ticks")
    render_side_legends(cases, overwide=True, axes_mode="all_axes")


def render_numbered(cases):
    """Draw a second collage with a separate species key for every panel."""
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_CM / 2.54, NUMBERED_FIGURE_HEIGHT_CM / 2.54)
    )
    outer = fig.add_gridspec(
        2, 1, left=0.10, right=0.985, bottom=0.065, top=0.97,
        height_ratios=(11.3, 9.7), hspace=0.16,
    )
    plots = outer[0].subgridspec(2, 3, wspace=0.13, hspace=0.24)
    keys = outer[1].subgridspec(
        2, 3, height_ratios=(6.9, 2.8), wspace=0.13, hspace=0.12
    )

    for index, (metal, plotter, stable_regions, labels) in enumerate(cases):
        row, column = divmod(index, 3)
        ax = fig.add_subplot(plots[row, column])
        plotter.plot_stable_regions(
            stable_regions, labels,
            pH_exp_range=EXPERIMENTAL_PH_RANGE,
            V_exp_range=EXPERIMENTAL_POTENTIAL_RANGE,
            label_style="numbered", show_legend=False, ax=ax,
        )
        ax.set_title(f"({chr(97 + index)}) {metal}", loc="left", pad=2, fontsize=8)
        if row == 0:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if column > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        key_ax = fig.add_subplot(keys[row, column])
        key_ax.set_axis_off()
        key_ax.text(
            0.02, 0.99, f"({chr(97 + index)}) {metal}",
            transform=key_ax.transAxes, ha="left", va="top", fontsize=7,
        )
        phase_labels = plotter.phase_key
        step = 0.89 / max(len(phase_labels), 1)
        for number, label in enumerate(phase_labels, start=1):
            y = 0.91 - (number - 0.5) * step
            key_ax.text(
                0.06, y, str(number), transform=key_ax.transAxes,
                ha="center", va="center", fontsize=6.5,
                bbox={"boxstyle": "circle,pad=0.18", "facecolor": "none",
                      "edgecolor": "black", "linewidth": 0.4},
            )
            key_ax.text(
                0.16, y, label, transform=key_ax.transAxes,
                ha="left", va="center", fontsize=6.5,
            )

    # add_phase_family_key(fig)
    for extension in ("png", "svg", "pdf", "tif"):
        output_path = OUTPUT_DIR / f"{OUTPUT_STEM}_numbered.{extension}"
        fig.savefig(
            output_path,
            format="tiff" if extension == "tif" else extension,
            dpi=DPI,
        )
        print(f"saved {output_path}")
    plt.close(fig)


def render_side_legends(
    cases, columns=3, overwide=False, axes_mode="minimal",
    colored_only=False, output_dir=None, output_stem=None,
):
    """Place a numbered or colored species list beside each square panel."""
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    output_stem = OUTPUT_STEM if output_stem is None else output_stem
    if axes_mode not in {"minimal", "upper_ticks", "all_axes"}:
        raise ValueError("Unknown axis-label mode")
    if axes_mode != "minimal" and not overwide:
        raise ValueError("Extra axis-label modes require the overwide layout")
    if overwide:
        if columns != 3:
            raise ValueError("The overwide side-legend layout requires three columns")
        if axes_mode == "all_axes":
            figure_width_cm = TWO_BY_THREE_FULL_AXES_WIDTH_CM
            figure_height_cm = TWO_BY_THREE_FULL_AXES_HEIGHT_CM
            row_bottom_cm = (7.4, 1.0)
        elif axes_mode == "upper_ticks":
            figure_width_cm = TWO_BY_THREE_OVERWIDE_FIGURE_WIDTH_CM
            figure_height_cm = 12.5
            row_bottom_cm = (6.8, 1.0)
        else:
            figure_width_cm = TWO_BY_THREE_OVERWIDE_FIGURE_WIDTH_CM
            figure_height_cm = TWO_BY_THREE_OVERWIDE_FIGURE_HEIGHT_CM
            row_bottom_cm = (6.3, 1.0)
        plot_width_cm = LARGE_PANEL_CM
        plot_left_offset_cm = 1.00
        key_left_offset_cm = 5.85
        key_right_margin_cm = 0.10
        key_height_cm = 5.2
        output_prefix = "2x3_overwide_"
    elif columns == 3:
        figure_width_cm = FIGURE_WIDTH_CM
        figure_height_cm = SIDE_KEY_FIGURE_HEIGHT_CM
        row_bottom_cm = (6.8, 0.6)
        plot_width_cm = 2.65
        plot_left_offset_cm = 0.70
        key_left_offset_cm = 3.48
        key_right_margin_cm = 0.07
        key_height_cm = 6.0
        output_prefix = ""
    elif columns == 2:
        figure_width_cm = THREE_BY_TWO_FIGURE_WIDTH_CM
        if len(cases) == 2:
            figure_height_cm = 6.8
            row_bottom_cm = (0.8,)
            output_prefix = "1x2_"
        elif len(cases) == 6:
            figure_height_cm = THREE_BY_TWO_FIGURE_HEIGHT_CM
            row_bottom_cm = (11.25, 6.0, 0.8)
            output_prefix = "3x2_"
        else:
            raise ValueError("Two-column side legends require two or six cases")
        plot_width_cm = LARGE_PANEL_CM
        plot_left_offset_cm = 1.00
        key_left_offset_cm = 5.85
        key_right_margin_cm = 0.10
        key_height_cm = 5.2
    else:
        raise ValueError("Side legends support only two or three columns")

    column_width_cm = FIGURE_WIDTH_CM / (2 if overwide else columns)
    column_stride_cm = (
        8.65 if overwide and axes_mode == "all_axes"
        else column_width_cm if columns == 3 and not overwide else 7.65
    )
    plot_height_cm = plot_width_cm

    label_styles = (("legend", "side_colors"),) if overwide else (
        ("numbered", "side_keys"), ("legend", "side_colors")
    )
    if colored_only:
        label_styles = (("legend", "side_colors"),)
    for label_style, suffix in label_styles:
        fig = plt.figure(
            figsize=(figure_width_cm / 2.54, figure_height_cm / 2.54)
        )
        for index, (metal, plotter, stable_regions, labels) in enumerate(cases):
            row, column = divmod(index, columns)
            column_left_cm = column * column_stride_cm
            plot_left_cm = column_left_cm + plot_left_offset_cm
            plot_bottom_cm = row_bottom_cm[row] + (key_height_cm - plot_height_cm) / 2
            ax = fig.add_axes((
                plot_left_cm / figure_width_cm,
                plot_bottom_cm / figure_height_cm,
                plot_width_cm / figure_width_cm,
                plot_height_cm / figure_height_cm,
            ))
            plotter.plot_stable_regions(
                stable_regions, labels,
                pH_exp_range=EXPERIMENTAL_PH_RANGE,
                V_exp_range=EXPERIMENTAL_POTENTIAL_RANGE,
                label_style=label_style, show_legend=False, ax=ax,
                legend_category_order=(
                    OVERWIDE_LEGEND_CATEGORY_ORDER
                    if overwide and axes_mode == "minimal" else None
                ),
            )
            if overwide and index == 0:
                annotate_au_water_lines(ax)
            ax.set_title(
                f"({chr(97 + index)}) {metal}", loc="left",
                pad=5 if columns == 2 or overwide else 2, fontsize=8,
            )
            if axes_mode == "all_axes":
                ax.yaxis.labelpad = 1
            else:
                ax.set_ylabel("")
            if row < len(row_bottom_cm) - 1 and axes_mode != "all_axes":
                ax.set_xlabel("")
                if axes_mode == "minimal":
                    ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if column > 0 and axes_mode == "minimal":
                ax.tick_params(axis="y", labelleft=False)

            key_left_cm = column_left_cm + key_left_offset_cm
            key_width_cm = column_width_cm - key_left_offset_cm - key_right_margin_cm
            key_ax = fig.add_axes((
                key_left_cm / figure_width_cm,
                row_bottom_cm[row] / figure_height_cm,
                key_width_cm / figure_width_cm,
                key_height_cm / figure_height_cm,
            ))
            key_ax.set_axis_off()
            phase_entries = (
                tuple((label, None) for label in plotter.phase_key)
                if label_style == "numbered" else plotter.color_phase_key
            )
            step = min(0.083, 0.88 / max(len(phase_entries), 1))
            first_y = 0.5 + (len(phase_entries) - 1) * step / 2
            for number, (label, color) in enumerate(phase_entries, start=1):
                y = first_y - (number - 1) * step
                if label_style == "numbered":
                    key_ax.text(
                        0.09, y, str(number), transform=key_ax.transAxes,
                        ha="center", va="center", fontsize=7,
                        bbox={"boxstyle": "circle,pad=0.16", "facecolor": "none",
                              "edgecolor": "black", "linewidth": 0.45},
                    )
                else:
                    alpha = plotter.style.phase_image_alpha
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

        if axes_mode != "all_axes":
            for row_bottom in row_bottom_cm:
                fig.text(
                    0.015,
                    (row_bottom + key_height_cm / 2) / figure_height_cm,
                    r"$E_{\mathrm{SHE}}$ (V)",
                    rotation=90, ha="center", va="center", fontsize=8,
                )
        output_suffix = "" if axes_mode == "minimal" else f"_{axes_mode}"
        for extension in ("png", "svg", "pdf", "tif"):
            output_path = output_dir / (
                f"{output_stem}_{output_prefix}{suffix}{output_suffix}.{extension}"
            )
            fig.savefig(
                output_path,
                format="tiff" if extension == "tif" else extension,
                dpi=DPI,
            )
            print(f"saved {output_path}")
        plt.close(fig)


def render_two_by_three_below_legends(cases, axes_mode="minimal"):
    """Keep square plots and below-panel keys with selectable axis detail."""
    if axes_mode == "all_axes":
        figure_height_cm = 20.0
        plot_bottom_cm = (14.65, 4.7)
        key_bottom_cm = (10.55, 0.35)
    elif axes_mode == "upper_ticks":
        figure_height_cm = TWO_BY_THREE_BELOW_FIGURE_HEIGHT_CM
        plot_bottom_cm = (12.65, 4.2)
        key_bottom_cm = (9.1, 0.35)
    elif axes_mode == "minimal":
        figure_height_cm = TWO_BY_THREE_BELOW_FIGURE_HEIGHT_CM
        plot_bottom_cm = (12.65, 4.2)
        key_bottom_cm = (9.4, 0.35)
    else:
        raise ValueError("Unknown axis-label mode")

    plot_width_cm = LARGE_PANEL_CM
    column_stride_cm = FIGURE_WIDTH_CM / 3
    plot_left_offset_cm = 0.9
    key_height_cm = 3.0
    key_left_offset_cm = 0.1
    key_width_cm = 5.5
    fig = plt.figure(figsize=(FIGURE_WIDTH_CM / 2.54, figure_height_cm / 2.54))

    for index, (metal, plotter, stable_regions, labels) in enumerate(cases):
        row, column = divmod(index, 3)
        column_left_cm = column * column_stride_cm
        ax = fig.add_axes((
            (column_left_cm + plot_left_offset_cm) / FIGURE_WIDTH_CM,
            plot_bottom_cm[row] / figure_height_cm,
            plot_width_cm / FIGURE_WIDTH_CM,
            plot_width_cm / figure_height_cm,
        ))
        plotter.plot_stable_regions(
            stable_regions, labels,
            pH_exp_range=EXPERIMENTAL_PH_RANGE,
            V_exp_range=EXPERIMENTAL_POTENTIAL_RANGE,
            label_style="legend", show_legend=False, ax=ax,
        )
        ax.set_title(f"({chr(97 + index)}) {metal}", loc="left", pad=5, fontsize=8)
        if axes_mode == "all_axes":
            ax.yaxis.labelpad = 1
        else:
            ax.set_ylabel("")
        if row == 0 and axes_mode != "all_axes":
            ax.set_xlabel("")
            if axes_mode == "minimal":
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
        if column > 0 and axes_mode == "minimal":
            ax.tick_params(axis="y", labelleft=False)

        key_ax = fig.add_axes((
            (column_left_cm + key_left_offset_cm) / FIGURE_WIDTH_CM,
            key_bottom_cm[row] / figure_height_cm,
            key_width_cm / FIGURE_WIDTH_CM,
            key_height_cm / figure_height_cm,
        ))
        key_ax.set_axis_off()
        phase_entries = plotter.color_phase_key
        rows_per_key = (len(phase_entries) + 1) // 2
        step = min(0.13, 0.84 / max(rows_per_key, 1))
        first_y = 0.5 + (rows_per_key - 1) * step / 2
        for entry_index, (label, color) in enumerate(phase_entries):
            key_column, key_row = divmod(entry_index, rows_per_key)
            x = 0.04 + 0.5 * key_column
            y = first_y - key_row * step
            alpha = plotter.style.phase_image_alpha
            blended_color = tuple(
                alpha * channel + (1 - alpha) for channel in color[:3]
            )
            key_ax.plot(
                x, y, marker="o", markersize=5,
                markerfacecolor=blended_color, markeredgecolor="none",
                linestyle="none", transform=key_ax.transAxes,
            )
            key_ax.text(
                x + 0.055, y, label, transform=key_ax.transAxes,
                ha="left", va="center", fontsize=7,
            )

    if axes_mode != "all_axes":
        for bottom in plot_bottom_cm:
            fig.text(
                0.015, (bottom + plot_width_cm / 2) / figure_height_cm,
                r"$E_{\mathrm{SHE}}$ (V)", rotation=90,
                ha="center", va="center", fontsize=8,
            )
    output_suffix = "" if axes_mode == "minimal" else f"_{axes_mode}"
    for extension in ("png", "svg", "pdf", "tif"):
        output_path = OUTPUT_DIR / (
            f"{OUTPUT_STEM}_2x3_below_colors{output_suffix}.{extension}"
        )
        fig.savefig(
            output_path,
            format="tiff" if extension == "tif" else extension,
            dpi=DPI,
        )
        print(f"saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-overwide-side-colors", action="store_true",
        help="Regenerate only the 2-by-3 overwide side-color figure.",
    )
    main(only_overwide_side_colors=parser.parse_args().only_overwide_side_colors)
