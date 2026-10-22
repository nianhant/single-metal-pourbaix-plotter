"""Generate Pourbaix diagrams over metal, activity, temperature, and ligand grids."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "methods"))

from data import Data
from grid_plotter import GridPlotter
from metal_complex_dataloader import MetalComplexDataLoader
from species import Species
from stability_calculator import StabilityCalculator
from thermodynamics import Thermodynamics


MU_LIGAND = {
    "NH3": -0.276037,
    "Gly": -3.263014109,
    "CN": 1.786800089,
    "NO2": -0.333729483,
}
LIGAND_PRESETS = {
    "baseline": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.1, "CN": 1e-4},
    "aqueous_only": {"NH3": 0.0, "NO2": 0.0, "Gly": 0.0, "CN": 0.0},
    "nh3_gly_low": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.005, "CN": 0.0},
    "nh3_gly_cn_low": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.005, "CN": 1e-4},
    "nh3_gly_mid": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.05, "CN": 0.0},
    "nh3_gly_cn_mid": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.05, "CN": 1e-4},
    "nh3_gly_0p1": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.1, "CN": 0.0},
    "nh3_gly_cn_0p1": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.1, "CN": 1e-4},
}
LEGACY_COMPLEX_OVERRIDES = {
    "Pd": {"Pd(CN)4[2-]": 6.467825128},
}


def load_complex_dataframe():
    """Load charge-balanced complex formulas and display labels."""
    json_path = ROOT / "data" / "metal_complex_del_G.json"
    return MetalComplexDataLoader(str(json_path)).load()


def parse_ligand_sets(value):
    if not value:
        return [LIGAND_PRESETS["aqueous_only"]]
    if value.lstrip().startswith("["):
        sets = json.loads(value)
    else:
        sets = [LIGAND_PRESETS[name] for name in value.split(",")]
    required = set(MU_LIGAND)
    if any(set(item) != required for item in sets):
        raise ValueError(f"Ligand sets must contain exactly {sorted(required)}")
    return sets


def load_energies(metal):
    data_dir = ROOT / "examples" / "data"
    with (data_dir / f"{metal}_solid_formation_energy.json").open() as handle:
        solid = json.load(handle)
    with (data_dir / f"{metal}_ion_formation_energy.json").open() as handle:
        aqueous = json.load(handle)
    aqueous.pop("FeOH[2+]", None)
    return solid, aqueous


def prepare_case(
    metal,
    activity,
    temperature,
    ligand_concentration,
    complex_df,
    ligand_correction,
    use_legacy_overrides=True,
):
    target = complex_df[complex_df["metal"] == metal].copy()
    if ligand_correction:
        target["energy"] = target["del_G_eV"] + (
            target["G_ligand (kJ/mol)"] / 96.485 * target["n_complex"]
        )
    else:
        target["energy"] = target["del_G_eV"]

    complexes = target.set_index("species")["energy"].to_dict()
    labels = target.set_index("species")["species_label"].to_dict()
    if use_legacy_overrides and metal in LEGACY_COMPLEX_OVERRIDES:
        complexes.update(LEGACY_COMPLEX_OVERRIDES[metal])
    solid, aqueous = load_energies(metal)
    data = Data(
        metal=metal,
        mu_bulk=solid,
        mu_aqueous_ion=aqueous,
        mu_complex=complexes,
        mu_ligand=MU_LIGAND,
        ligand_concentration=ligand_concentration,
        ion_activity=activity,
        T=temperature,
    )
    thermo = Thermodynamics(T=temperature)
    species = []
    for phase, potentials in (("bulk", solid), ("aqueous_ion", aqueous), ("complex", complexes)):
        for formula in potentials:
            species.append(Species(
                formula=formula,
                metal=metal,
                thermo=thermo,
                data=data,
                phase=phase,
                activity=0 if phase == "bulk" else activity,
            ))
    return data, thermo, species, labels


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--metals", nargs="+", default=["Cu"])
    result.add_argument("--activities", nargs="+", type=float, default=[1e-4])
    result.add_argument("--temperatures", nargs="+", type=float, default=[298.15])
    result.add_argument("--ligand-sets", help="Preset names separated by commas, or a JSON list")
    result.add_argument(
        "--ligand-correction",
        action="store_true",
        help=(
            "Apply the legacy extra ligand-reference shift. Normally leave this disabled: "
            "del_G_eV already includes the free-ligand standard Gibbs energies."
        ),
    )
    result.add_argument(
        "--tabulated-pd-cyanide-energy",
        action="store_false",
        dest="use_legacy_overrides",
        help=(
            "Use the tabulated Pd(CN)4[2-] energy instead of the historical "
            "energy used by the archived Pourbaix diagrams."
        ),
    )
    result.set_defaults(use_legacy_overrides=True)
    result.add_argument("--grid-size", type=int, default=400)
    result.add_argument("--ph-range", nargs=2, type=float, default=(-2.0, 16.0))
    result.add_argument("--potential-range", nargs=2, type=float, default=(-2.0, 3.0))
    result.add_argument("--box-ph-range", nargs=2, type=float, default=(10.5, 13.5))
    result.add_argument("--box-potential-range", nargs=2, type=float, default=(-2.0, 2.3))
    result.add_argument("--formats", nargs="+", default=["svg", "pdf", "tif"])
    result.add_argument("--label-style", choices=["direct", "numbered", "legend"], default="direct")
    result.add_argument("--dpi", type=int, default=600)
    result.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "examples" / "figures" / "figures" / "batch_pourbaix",
    )
    return result


def main():
    args = parser().parse_args()
    ligand_sets = parse_ligand_sets(args.ligand_sets)
    complex_df = load_complex_dataframe()

    for metal in args.metals:
        for activity in args.activities:
            for temperature in args.temperatures:
                for ligand_concentration in ligand_sets:
                    data, thermo, species, labels = prepare_case(
                        metal,
                        activity,
                        temperature,
                        ligand_concentration,
                        complex_df,
                        args.ligand_correction,
                        args.use_legacy_overrides,
                    )
                    filename = (
                        f"{metal}-NH3-H2O_T={temperature:.2f}_activity={activity:.0e}"
                        f"_NH3={ligand_concentration['NH3']}"
                        f"_Gly={ligand_concentration['Gly']}"
                        f"_CN={ligand_concentration['CN']}"
                    )
                    if args.label_style in {"numbered", "legend"}:
                        filename += f"_{args.label_style}"
                    plotter = GridPlotter(
                        pH_range=tuple(args.ph_range),
                        V_range=tuple(args.potential_range),
                        data=data,
                        grid_size=args.grid_size,
                        save_fig=True,
                        dir=str(args.output_root.resolve() / metal),
                        filename=filename,
                        output_formats=args.formats,
                        dpi=args.dpi,
                    )
                    stable_regions = StabilityCalculator(
                        plotter, species, thermo, data
                    ).compute_stable_regions()
                    plotter.plot_stable_regions(
                        stable_regions,
                        labels,
                        avoid_label_overlap=args.label_style == "direct",
                        label_indicator="line",
                        pH_exp_range=tuple(args.box_ph_range),
                        V_exp_range=tuple(args.box_potential_range),
                        label_style=args.label_style,
                    )


if __name__ == "__main__":
    main()
