"""Run one-at-a-time Pourbaix sensitivity analysis around manuscript baselines."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "methods"))

from grid_plotter import (  # noqa: E402
    CM_PER_INCH,
    DEFAULT_PUBLICATION_STYLE,
    GridPlotter,
    RSC_DOUBLE_COLUMN_WIDTH_CM,
)
from stability_calculator import StabilityCalculator  # noqa: E402

from generate_batch_pourbaix import (  # noqa: E402
    load_complex_dataframe,
    prepare_case,
)


METALS = ("Cu", "Au", "Ni", "Pt", "Pd", "Ti")
TEMPERATURE_K = 298.15
BASELINE = {
    "NH3": 2e-2,
    "Gly": 1e-1,
    "CN": 1e-4,
    "activity": 1e-4,
}
SENSITIVITY_LEVELS = {
    "NH3": {"low": 2e-3, "baseline": BASELINE["NH3"], "high": 1e-1},
    "Gly": {"low": 1e-2, "baseline": BASELINE["Gly"], "high": 1.0},
    "CN": {"low": 1e-6, "baseline": BASELINE["CN"], "high": 1e-3},
    "activity": {"low": 1e-6, "baseline": BASELINE["activity"], "high": 1e-2},
}
CATEGORIES = (
    "metal",
    "metal_hydride",
    "metal_oxide",
    "metal_oxyhydroxide",
    "aqueous_metal_ion",
    "metal_ligand_complex",
)
PARAMETER_LABELS = {
    "NH3": r"$[\mathrm{NH_3}]_{\mathrm{tot}}$ (M)",
    "Gly": r"$[\mathrm{Gly}]_{\mathrm{tot}}$ (M)",
    "CN": r"$[\mathrm{CN}]_{\mathrm{tot}}$ (M)",
    "activity": "Dissolved-ion activity (M)",
}


def parser():
    """Build the command-line parser for the sensitivity workflow."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--metals", nargs="+", default=list(METALS))
    result.add_argument("--grid-size", type=int, default=800)
    result.add_argument("--formats", nargs="+", default=["png", "pdf"])
    result.add_argument("--dpi", type=int, default=600)
    result.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT / "examples" / "figures" / "figures"
            / "sensitivity_analysis"
        ),
    )
    return result


def condition_for(parameter, value):
    """Return ligand concentrations and ion activity for one OAT condition."""
    condition = dict(BASELINE)
    condition[parameter] = value
    ligands = {
        "NH3": condition["NH3"],
        "NO2": 0.0,
        "Gly": condition["Gly"],
        "CN": condition["CN"],
    }
    return ligands, condition["activity"]


def filename_for(metal, parameter, level, value, ligands, activity):
    """Build a descriptive filename without the unused NO2 concentration."""
    return (
        f"{metal}_sensitivity_{parameter}_{level}={value:.0e}"
        f"_activity={activity:.0e}_NH3={ligands['NH3']:.0e}"
        f"_Gly={ligands['Gly']:.0e}_CN={ligands['CN']:.0e}"
    )


def category_fractions(plotter, stable_regions):
    """Compute normalized pH-potential area fractions by phase category."""
    fractions = {category: 0.0 for category in CATEGORIES}
    total_points = plotter.pH_grid.size
    for product, stable_mask in stable_regions.items():
        fractions[plotter.classify_product(product)] += (
            np.count_nonzero(stable_mask) / total_points
        )
    fractions["aqueous_total"] = (
        fractions["aqueous_metal_ion"] + fractions["metal_ligand_complex"]
    )
    return fractions


def write_summary(records, output_root):
    """Write machine-readable sensitivity results."""
    output_path = output_root / "sensitivity_phase_area_fractions.csv"
    fieldnames = [
        "parameter",
        "level",
        "value",
        "element",
        *CATEGORIES,
        "aqueous_total",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def plot_summary(records, output_root, dpi):
    """Plot aqueous-domain fraction versus each varied parameter."""
    style = DEFAULT_PUBLICATION_STYLE
    rc_params = {
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.tick_label_font_size,
        "axes.labelsize": style.axis_label_font_size,
        "axes.titlesize": style.axis_label_font_size,
        "xtick.labelsize": style.tick_label_font_size,
        "ytick.labelsize": style.tick_label_font_size,
        "legend.fontsize": style.legend_label_font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    figure_height_cm = 11.3
    with plt.rc_context(rc_params):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(
                RSC_DOUBLE_COLUMN_WIDTH_CM / CM_PER_INCH,
                figure_height_cm / CM_PER_INCH,
            ),
            constrained_layout=True,
        )
        for axis, parameter in zip(axes.flat, SENSITIVITY_LEVELS):
            for metal in METALS:
                selected = sorted(
                    (
                        record for record in records
                        if record["parameter"] == parameter
                        and record["element"] == metal
                    ),
                    key=lambda record: record["value"],
                )
                if not selected:
                    continue
                axis.plot(
                    [record["value"] for record in selected],
                    [record["aqueous_total"] for record in selected],
                    marker="o",
                    markersize=3.5,
                    linewidth=0.8,
                    label=metal,
                )
            axis.set_xscale("log")
            axis.set_xlabel(PARAMETER_LABELS[parameter])
            axis.set_ylabel("Aqueous-domain fraction")
            axis.set_ylim(0.0, 1.0)
            axis.grid(alpha=0.25, linewidth=0.4)

        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=len(labels),
            frameon=False,
        )
        for extension in ("png", "pdf", "tif"):
            output_path = output_root / f"sensitivity_aqueous_domain_fraction.{extension}"
            save_format = "tiff" if extension == "tif" else extension
            figure.savefig(output_path, format=save_format, dpi=dpi)
        plt.close(figure)


def main():
    """Generate sensitivity diagrams, phase fractions, and summary plots."""
    args = parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    complex_df = load_complex_dataframe()
    records = []

    for parameter, levels in SENSITIVITY_LEVELS.items():
        for level, value in levels.items():
            ligands, activity = condition_for(parameter, value)
            for metal in args.metals:
                data, thermo, species, labels = prepare_case(
                    metal,
                    activity,
                    TEMPERATURE_K,
                    ligands,
                    complex_df,
                    ligand_correction=False,
                    use_legacy_overrides=True,
                )
                output_dir = args.output_root / parameter / level / metal
                filename = filename_for(
                    metal, parameter, level, value, ligands, activity
                )
                plotter = GridPlotter(
                    pH_range=(-2.0, 16.0),
                    V_range=(-2.0, 3.0),
                    data=data,
                    grid_size=args.grid_size,
                    save_fig=True,
                    dir=str(output_dir),
                    filename=filename,
                    output_formats=args.formats,
                    dpi=args.dpi,
                )
                stable_regions = StabilityCalculator(
                    plotter, species, thermo, data
                ).compute_stable_regions()
                fractions = category_fractions(plotter, stable_regions)
                records.append({
                    "parameter": parameter,
                    "level": level,
                    "value": value,
                    "element": metal,
                    **fractions,
                })
                plotter.plot_stable_regions(
                    stable_regions,
                    labels,
                    avoid_label_overlap=True,
                    label_indicator="line",
                    pH_exp_range=(11.5, 13.5),
                    V_exp_range=(-2.0, 2.3),
                    label_style="direct",
                )

    write_summary(records, args.output_root)
    plot_summary(records, args.output_root, args.dpi)


if __name__ == "__main__":
    main()
