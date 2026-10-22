"""Generate a Pd/Pt cyanide Pourbaix panel with six-panel side-legend styling.

Run ``python examples/generate_pd_pt_cn_panel.py`` from the repository root.
"""

from generate_batch_pourbaix import (
    GridPlotter,
    LIGAND_PRESETS,
    ROOT,
    StabilityCalculator,
    load_complex_dataframe,
    prepare_case,
)
from generate_six_panel_pourbaix import (
    ACTIVITY,
    DPI,
    GRID_SIZE,
    PH_RANGE,
    POTENTIAL_RANGE,
    TEMPERATURE,
    render_side_legends,
)


METALS = ("Pd", "Pt")
LIGAND_CONCENTRATIONS = LIGAND_PRESETS["nh3_gly_cn_0p1"]
OUTPUT_DIR = ROOT / "examples" / "figures" / "figures" / "pd_pt_nh3_gly_0p1_cn_1e-4"
OUTPUT_STEM = "Pd_Pt_activity_1e-4_NH3_0p02_Gly_0p1_CN_1e-4"


def main() -> None:
    """Compute both stability maps at GRID_SIZE and save the paired panel."""
    complex_df = load_complex_dataframe()
    cases = []
    for metal in METALS:
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_side_legends(
        cases,
        columns=2,
        colored_only=True,
        output_dir=OUTPUT_DIR,
        output_stem=OUTPUT_STEM,
    )


if __name__ == "__main__":
    main()
