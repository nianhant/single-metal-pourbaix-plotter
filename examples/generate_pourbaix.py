import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "methods"
if str(METHODS_DIR) not in sys.path:
    sys.path.append(str(METHODS_DIR))

from data import Data
from grid_plotter import GridPlotter
from metal_complex_dataloader import MetalComplexDataLoader
from pymatgen_api import (
    get_ion_formation_energy,
    get_mpr,
    get_solid_formation_energy,
)
from species import Species
from stability_calculator import StabilityCalculator
from thermodynamics import Thermodynamics

MU_LIGAND = {
    "NH3": -0.276037,
    "Gly": -3.263014109,
    "CN": 1.786800089,
    "NO2": -0.333729483,
}


DEFAULTS = {
    "metal": "Ni",
    "temperature": 298.15,
    "activity": 1e-4,
    "nh3": 0.02,
    "no2": 0.0,
    "gly": 0.005,
    "cn": 0.0,
    "grid_size": 400,
    "pH_range": (-2.0, 16.0),
    "V_range": (-2.0, 3.0),
    "output_dir": "figures/pourbaix_diagrams",
}


@dataclass(frozen=True)
class DiagramSettings:
    metal: str
    temperature: float
    activity: float
    ligand_concentration: dict
    grid_size: int
    pH_range: tuple
    V_range: tuple
    output_dir: Path


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


def ligand_case_label(ligand_concentration):
    return "_".join(
        f"{ligand}={value:g}M"
        for ligand, value in ligand_concentration.items()
    )


def output_filename(settings):
    return (
        f"{settings.metal}-NH3-H2O_T={settings.temperature:.2f}"
        f"_activity={settings.activity:.0e}_"
        f"{ligand_case_label(settings.ligand_concentration)}.png"
    )


def load_complex_energies(metal):
    loader = MetalComplexDataLoader(str(ROOT / "data" / "metal_complex_del_G.json"))
    complex_df = loader.load()
    target_df = complex_df[complex_df["metal"] == metal].copy()
    return (
        target_df.set_index("species")["del_G_eV"].to_dict(),
        target_df.set_index("species")["species_label"].to_dict(),
    )


def load_energy_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_energy_json(path: Path, energy_data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(energy_data, handle, indent=4)


def load_formation_energies(metal):
    solid_path = ROOT / "data" / f"{metal}_solid_formation_energy.json"
    ion_path = ROOT / "data" / f"{metal}_ion_formation_energy.json"
    if not solid_path.exists() or not ion_path.exists():
        print(
            f"Missing cached formation-energy JSON files for {metal}; "
            "fetching from Materials Project."
        )
        mpr = get_mpr()
        if not solid_path.exists():
            solid_eng = get_solid_formation_energy(metal, mpr, save_json=False)
            save_energy_json(solid_path, solid_eng)
        if not ion_path.exists():
            ion_eng = get_ion_formation_energy(metal, mpr, save_json=False)
            save_energy_json(ion_path, ion_eng)

    solid_eng = load_energy_json(solid_path)
    ion_eng = load_energy_json(ion_path)
    ion_eng.pop("FeOH[2+]", None)
    return solid_eng, ion_eng


def build_species(settings, thermo, metal_data, solid_eng, ion_eng, metal_complex):
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
                    metal=settings.metal,
                    thermo=thermo,
                    data=metal_data,
                    phase=phase,
                    activity=phase_activity,
                )
            )
    return all_species


def make_diagram(settings):
    solid_eng, ion_eng = load_formation_energies(settings.metal)
    metal_complex, species_label_dict = load_complex_energies(settings.metal)
    thermo = Thermodynamics(T=settings.temperature)
    metal_data = Data(
        metal=settings.metal,
        mu_bulk=solid_eng,
        mu_aqueous_ion=ion_eng,
        mu_complex=metal_complex,
        mu_ligand=MU_LIGAND,
        ligand_concentration=settings.ligand_concentration,
        ion_activity=settings.activity,
        T=settings.temperature,
    )
    all_species = build_species(settings, thermo, metal_data, solid_eng, ion_eng, metal_complex)

    plotter = GridPlotter(
        pH_range=settings.pH_range,
        V_range=settings.V_range,
        data=metal_data,
        grid_size=settings.grid_size,
        save_fig=True,
        dir=str(settings.output_dir / settings.metal),
        filename=output_filename(settings),
    )

    stable_regions = StabilityCalculator(plotter, all_species, thermo, metal_data).compute_stable_regions()
    plotter.plot_stable_regions(stable_regions, species_label_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one cached-data single-metal Pourbaix diagram."
    )
    parser.add_argument("--metal", default=DEFAULTS["metal"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--activity", type=float, default=DEFAULTS["activity"])
    parser.add_argument("--nh3", type=float, default=DEFAULTS["nh3"])
    parser.add_argument("--no2", type=float, default=DEFAULTS["no2"])
    parser.add_argument("--gly", type=float, default=DEFAULTS["gly"])
    parser.add_argument("--cn", type=float, default=DEFAULTS["cn"])
    parser.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    return parser.parse_args()


def build_settings(args):
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    return DiagramSettings(
        metal=args.metal,
        temperature=args.temperature,
        activity=args.activity,
        ligand_concentration={
            "NH3": args.nh3,
            "NO2": args.no2,
            "Gly": args.gly,
            "CN": args.cn,
        },
        grid_size=args.grid_size,
        pH_range=DEFAULTS["pH_range"],
        V_range=DEFAULTS["V_range"],
        output_dir=output_dir,
    )


def main():
    set_publication_style()
    make_diagram(build_settings(parse_args()))


if __name__ == "__main__":
    main()
