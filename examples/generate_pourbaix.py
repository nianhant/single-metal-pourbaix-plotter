import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "methods"
if str(METHODS_DIR) not in sys.path:
    sys.path.append(str(METHODS_DIR))

from data import Data
from grid_plotter import GridPlotter
from metal_complex_dataloader import MetalComplexDataLoader
from species import Species
from stability_calculator import StabilityCalculator
from thermodynamics import Thermodynamics

METAL = "Ni"
TEMPERATURE_K = 298.15
ACTIVITY = 1e-4
LIGAND_CONCENTRATION = {"NH3": 0.02, "NO2": 0.0, "Gly": 0.005, "CN": 0.0}
MU_LIGAND = {
    "NH3": -0.276037,
    "Gly": -3.263014109,
    "CN": 1.786800089,
    "NO2": -0.333729483,
}

PH_RANGE = (-2.0, 16.0)
V_RANGE = (-2.0, 3.0)
GRID_SIZE = 400
OUTPUT_DIR = ROOT / "figures" / "pourbaix_diagrams" / METAL


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


def ligand_case_label():
    return "_".join(
        f"{ligand}={value:g}M"
        for ligand, value in LIGAND_CONCENTRATION.items()
    )


def output_filename():
    return (
        f"{METAL}-NH3-H2O_T={TEMPERATURE_K:.2f}"
        f"_activity={ACTIVITY:.0e}_{ligand_case_label()}.png"
    )


def load_complex_energies():
    loader = MetalComplexDataLoader(str(ROOT / "data" / "metal_complex_del_G.json"))
    complex_df = loader.load()
    target_df = complex_df[complex_df["metal"] == METAL].copy()
    return (
        target_df.set_index("species")["del_G_eV"].to_dict(),
        target_df.set_index("species")["species_label"].to_dict(),
    )


def load_energy_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_formation_energies():
    solid_eng = load_energy_json(ROOT / "data" / f"{METAL}_solid_formation_energy.json")
    ion_eng = load_energy_json(ROOT / "data" / f"{METAL}_ion_formation_energy.json")
    ion_eng.pop("FeOH[2+]", None)
    return solid_eng, ion_eng


def build_species(thermo, metal_data, solid_eng, ion_eng, metal_complex):
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
                    metal=METAL,
                    thermo=thermo,
                    data=metal_data,
                    phase=phase,
                    activity=phase_activity,
                )
            )
    return all_species


def main():
    set_publication_style()
    solid_eng, ion_eng = load_formation_energies()
    metal_complex, species_label_dict = load_complex_energies()

    metal_data = Data(
        metal=METAL,
        mu_bulk=solid_eng,
        mu_aqueous_ion=ion_eng,
        mu_complex=metal_complex,
        mu_ligand=MU_LIGAND,
        ligand_concentration=LIGAND_CONCENTRATION,
        ion_activity=ACTIVITY,
        T=TEMPERATURE_K,
    )
    thermo = Thermodynamics(T=TEMPERATURE_K)
    all_species = build_species(thermo, metal_data, solid_eng, ion_eng, metal_complex)

    plotter = GridPlotter(
        pH_range=PH_RANGE,
        V_range=V_RANGE,
        data=metal_data,
        grid_size=GRID_SIZE,
        save_fig=True,
        dir=str(OUTPUT_DIR),
        filename=output_filename(),
    )

    stable_regions = StabilityCalculator(plotter, all_species, thermo, metal_data).compute_stable_regions()
    plotter.plot_stable_regions(stable_regions, species_label_dict)


if __name__ == "__main__":
    main()
