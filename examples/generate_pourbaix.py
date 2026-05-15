import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "methods"
if str(METHODS_DIR) not in sys.path:
    sys.path.append(str(METHODS_DIR))

from data import Data
from grid_plotter import GridPlotter
from metal_complex_dataloader import MetalComplexDataLoader
from pymatgen_api import get_ion_formation_energy, get_mpr, get_solid_formation_energy
from species import Species
from stability_calculator import StabilityCalculator
from thermodynamics import Thermodynamics

DEFAULT_MU_LIGAND = {
    "NH3": -0.276037,
    "Gly": -3.263014109,
    "CN": 1.786800089,
    "NO2": -0.333729483,
}

DEFAULT_LIGAND_PRESETS = {
    "aqueous_only": {"NH3": 0.0, "NO2": 0.0, "Gly": 0.0, "CN": 0.0},
    "gly_only": {"NH3": 0.0, "NO2": 0.0, "Gly": 0.1, "CN": 0.0},
    "nh3_gly_low": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.005, "CN": 0.0},
    "nh3_gly_cn_low": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.005, "CN": 1e-4},
    "nh3_gly_mid": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.05, "CN": 0.0},
    "nh3_gly_cn_mid": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.05, "CN": 1e-4},
    "nh3_gly_high": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.1, "CN": 0.0},
    "nh3_gly_cn_high": {"NH3": 0.02, "NO2": 0.0, "Gly": 0.1, "CN": 1e-4},
    "experiment": {"NH3": 0.05, "NO2": 0.0, "Gly": 0.1, "CN": 0.001},
    "meng": {"NH3": 1.0, "NO2": 0.0, "Gly": 0.0, "CN": 0.0},
}

MISSING_COMPLEX_OVERRIDES = {
    "Pd": {"Pd(CN)4[2+]": 6.467825128},
    "Pt": {"Pt(CN)4[2+]": 5.646346039},
}


@dataclass(frozen=True)
class DiagramCase:
    metal: str
    activity: float
    temperature: float
    ligand_concentration: Dict[str, float]


@dataclass(frozen=True)
class PlotConfig:
    pH_range: Tuple[float, float]
    V_range: Tuple[float, float]
    grid_size: int
    save_fig: bool
    output_dir: str
    filename: Optional[str] = None


def set_publication_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 27,
            "axes.titlesize": 16,
            "xtick.labelsize": 27,
            "ytick.labelsize": 27,
            "legend.fontsize": 14,
            "figure.titlesize": 27,
            "savefig.dpi": 300,
        }
    )


def parse_float_list(values: Sequence[str]) -> List[float]:
    return [float(value) for value in values]


def build_ligand_concentration(
    nh3: float = 0.0,
    no2: float = 0.0,
    gly: float = 0.0,
    cn: float = 0.0,
) -> Dict[str, float]:
    return {"NH3": nh3, "NO2": no2, "Gly": gly, "CN": cn}


def ligand_case_label(ligand_concentration: Dict[str, float]) -> str:
    return "_".join(
        f"{ligand}={value:g}M"
        for ligand, value in ligand_concentration.items()
    )


def default_filename(case: DiagramCase) -> str:
    return (
        f"{case.metal}-H2O_T={case.temperature:.2f}"
        f"_activity={case.activity:.0e}_{ligand_case_label(case.ligand_concentration)}.png"
    )


def build_cases(
    metals: Iterable[str],
    activities: Iterable[float],
    temperatures: Iterable[float],
    ligand_sets: Iterable[Dict[str, float]],
) -> List[DiagramCase]:
    return [
        DiagramCase(
            metal=metal,
            activity=activity,
            temperature=temperature,
            ligand_concentration=dict(ligand_concentration),
        )
        for metal in metals
        for activity in activities
        for temperature in temperatures
        for ligand_concentration in ligand_sets
    ]


def load_complex_dataframe():
    loader = MetalComplexDataLoader(str(ROOT / "data" / "metal_complex_del_G.json"))
    df = loader.load()
    loader.save_to_csv(str(ROOT / "data" / "metal_complex_energies.csv"))
    return df


def load_energy_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_formation_energies(metal: str, mpr=None):
    solid_path = ROOT / "examples" / "data" / f"{metal}_solid_formation_energy.json"
    ion_path = ROOT / "examples" / "data" / f"{metal}_ion_formation_energy.json"

    if solid_path.exists():
        solid_eng = load_energy_json(solid_path)
    else:
        if mpr is None:
            raise ValueError(f"Missing cached solid energies for {metal} and no MP API client available.")
        solid_eng = get_solid_formation_energy(metal, mpr)

    if ion_path.exists():
        ion_eng = load_energy_json(ion_path)
    else:
        if mpr is None:
            raise ValueError(f"Missing cached ion energies for {metal} and no MP API client available.")
        ion_eng = get_ion_formation_energy(metal, mpr)

    ion_eng.pop("FeOH[2+]", None)
    return solid_eng, ion_eng


def build_complex_energy_dict(complex_df, metal: str, use_ligand_correction: bool):
    target_df = complex_df[complex_df["metal"] == metal].copy()
    if use_ligand_correction:
        ligand_mu_eV = target_df["G_ligand (kJ/mol)"] / 96.485
        target_df["complex_energy"] = target_df["del_G_eV"] + ligand_mu_eV * target_df["n_complex"]
    else:
        target_df["complex_energy"] = target_df["del_G_eV"]

    metal_complex = target_df.set_index("species")["complex_energy"].to_dict()
    metal_complex.update(MISSING_COMPLEX_OVERRIDES.get(metal, {}))
    species_label_dict = target_df.set_index("species")["species_label"].to_dict()
    return metal_complex, species_label_dict


def build_species(metal: str, thermo, metal_data, solid_eng, ion_eng, metal_complex):
    all_species = []
    for phase, chem_pot in (
        ("bulk", solid_eng),
        ("aqueous_ion", ion_eng),
        ("complex", metal_complex),
    ):
        phase_activity = 0 if phase == "bulk" else metal_data.ion_activity
        for formula in chem_pot:
            all_species.append(
                Species(
                    formula=formula,
                    metal=metal,
                    thermo=thermo,
                    data=metal_data,
                    phase=phase,
                    activity=phase_activity,
                )
            )
    return all_species


def run_case(case: DiagramCase, plot_config: PlotConfig, complex_df, mu_ligand, mpr=None, use_ligand_correction=False):
    solid_eng, ion_eng = load_formation_energies(case.metal, mpr=mpr)
    metal_complex, species_label_dict = build_complex_energy_dict(
        complex_df=complex_df,
        metal=case.metal,
        use_ligand_correction=use_ligand_correction,
    )

    metal_data = Data(
        metal=case.metal,
        mu_bulk=solid_eng,
        mu_aqueous_ion=ion_eng,
        mu_complex=metal_complex,
        mu_ligand=mu_ligand,
        ligand_concentration=case.ligand_concentration,
        ion_activity=case.activity,
        T=case.temperature,
    )
    thermo = Thermodynamics(T=case.temperature)
    all_species = build_species(case.metal, thermo, metal_data, solid_eng, ion_eng, metal_complex)

    plotter = GridPlotter(
        pH_range=plot_config.pH_range,
        V_range=plot_config.V_range,
        data=metal_data,
        grid_size=plot_config.grid_size,
        save_fig=plot_config.save_fig,
        dir=plot_config.output_dir.format(metal=case.metal),
        filename=plot_config.filename or default_filename(case),
    )

    stable_regions = StabilityCalculator(plotter, all_species, thermo, metal_data).compute_stable_regions()
    plotter.plot_stable_regions(stable_regions, species_label_dict)
    return stable_regions


def resolve_ligand_sets(args) -> List[Dict[str, float]]:
    if args.ligand_presets:
        return [dict(DEFAULT_LIGAND_PRESETS[name]) for name in args.ligand_presets]
    return [
        build_ligand_concentration(
            nh3=args.nh3,
            no2=args.no2,
            gly=args.gly,
            cn=args.cn,
        )
    ]


def build_parser():
    parser = argparse.ArgumentParser(description="Generate Pourbaix diagrams from a clean configurable entry point.")
    parser.add_argument("--metals", nargs="+", default=["Cu"])
    parser.add_argument("--activities", nargs="+", default=["1e-4"])
    parser.add_argument("--temperatures", nargs="+", default=["298.15"])
    parser.add_argument("--ligand-presets", nargs="*", choices=sorted(DEFAULT_LIGAND_PRESETS))
    parser.add_argument("--nh3", type=float, default=0.0)
    parser.add_argument("--no2", type=float, default=0.0)
    parser.add_argument("--gly", type=float, default=0.0)
    parser.add_argument("--cn", type=float, default=0.0)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--ph-min", type=float, default=-2.0)
    parser.add_argument("--ph-max", type=float, default=16.0)
    parser.add_argument("--v-min", type=float, default=-2.0)
    parser.add_argument("--v-max", type=float, default=3.0)
    parser.add_argument("--output-dir", default="figures/updated_pourbaix/{metal}")
    parser.add_argument("--no-save-fig", action="store_true")
    parser.add_argument("--use-ligand-correction", action="store_true")
    parser.add_argument("--mp-api-key")
    return parser


def main():
    set_publication_style()
    parser = build_parser()
    args = parser.parse_args()

    ligand_sets = resolve_ligand_sets(args)
    cases = build_cases(
        metals=args.metals,
        activities=parse_float_list(args.activities),
        temperatures=parse_float_list(args.temperatures),
        ligand_sets=ligand_sets,
    )

    plot_config = PlotConfig(
        pH_range=(args.ph_min, args.ph_max),
        V_range=(args.v_min, args.v_max),
        grid_size=args.grid_size,
        save_fig=not args.no_save_fig,
        output_dir=args.output_dir,
    )

    complex_df = load_complex_dataframe()
    mpr = None
    try:
        mpr = get_mpr(args.mp_api_key)
    except ValueError:
        mpr = None

    for case in cases:
        print(
            f"Generating {case.metal} at T={case.temperature:.2f} K, "
            f"activity={case.activity:.0e}, ligands={case.ligand_concentration}"
        )
        run_case(
            case=case,
            plot_config=plot_config,
            complex_df=complex_df,
            mu_ligand=DEFAULT_MU_LIGAND,
            mpr=mpr,
            use_ligand_correction=args.use_ligand_correction,
        )


if __name__ == "__main__":
    main()
