import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "methods"
if str(METHODS_DIR) not in sys.path:
    sys.path.append(str(METHODS_DIR))

from data import Data
from grid_plotter import GridPlotter
from metal_complex_dataloader import MetalComplexDataLoader
from reaction import Reaction
from species import Species
from stability_calculator import StabilityCalculator
from thermodynamics import Thermodynamics

ROOM_TEMPERATURE_K = 298.15
MU_LIGAND = {
    "NH3": -0.276037,
    "Gly": -3.263014109,
    "CN": 1.786800089,
    "NO2": -0.333729483,
}


def species_latex(formula: str, phase: str) -> str:
    suffix = "(s)" if phase == "bulk" else "(aq)"
    return f"\\ce{{{formula}}}{suffix}"


def coeff_str(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1e-8:
        value = rounded
    if value == 1:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value} "


def build_reaction_string(reaction: Reaction) -> str:
    mb = reaction.compute_mass_balance()
    reactants: List[Tuple[float, str]] = []
    products: List[Tuple[float, str]] = []

    reactants.append((mb["fraction"], species_latex(reaction.reactant.formula, reaction.reactant.phase)))
    # print(f"Reactant: {reaction.reactant.formula} ({reaction.reactant.phase})", f"product: {reaction.product.formula} ({reaction.product.phase})")
    # print(f"mass balance ligand: {mb['n_L']}")
    products.append((1, species_latex(reaction.product.formula, reaction.product.phase)))

    if mb["n_H"] > 0:
        reactants.append((mb["n_H"], r"\ce{H+}"))
    elif mb["n_H"] < 0:
        products.append((-mb["n_H"], r"\ce{H+}"))

    if mb["n_H2O"] > 0:
        products.append((mb["n_H2O"], r"\ce{H2O}"))
    elif mb["n_H2O"] < 0:
        reactants.append((-mb["n_H2O"], r"\ce{H2O}"))

    if mb["n_charge"] > 0:
        reactants.append((mb["n_charge"], r"\ce{e-}"))
    elif mb["n_charge"] < 0:
        products.append((-mb["n_charge"], r"\ce{e-}"))

    for ligand, count in mb["n_L"].items():
        ligand_tex = fr"\ce{{{ligand}}}"
        if count > 0:
            products.append((count, ligand_tex))
        elif count < 0:
            reactants.append((-count, ligand_tex))
    # if 'NH3' in mb['n_L']:
    #     print('reactants', reactants)
    #     print('products', products)

    left = " + ".join(f"{coeff_str(c)}{sp}" for c, sp in reactants)
    right = " + ".join(f"{coeff_str(c)}{sp}" for c, sp in products)
    return f"${left} \\rightarrow {right}$"


def touches(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(
        np.any(mask_a[:, :-1] & mask_b[:, 1:])
        or np.any(mask_a[:, 1:] & mask_b[:, :-1])
        or np.any(mask_a[:-1, :] & mask_b[1:, :])
        or np.any(mask_a[1:, :] & mask_b[:-1, :])
    )


def build_species_for_metal(
    metal: str,
    activity: float,
    ligand_conc: Dict[str, float],
    complex_df,
):
    thermo = Thermodynamics(T=ROOM_TEMPERATURE_K)

    with open(ROOT / "examples/data" / f"{metal}_solid_formation_energy.json", "r", encoding="utf-8") as f:
        solid_eng = json.load(f)
    with open(ROOT / "examples/data" / f"{metal}_ion_formation_energy.json", "r", encoding="utf-8") as f:
        ion_eng = json.load(f)

    target_df = complex_df[complex_df["metal"] == metal]
    complex_energies = target_df.set_index("species")["del_G_eV"].to_dict()

    if metal == "Pd":
        complex_energies.setdefault("Pd(CN)4[2+]", 6.467825128)
    if metal == "Pt":
        complex_energies.setdefault("Pt(CN)4[2+]", 5.646346039)

    metal_data = Data(
        metal=metal,
        mu_bulk=solid_eng,
        mu_aqueous_ion=ion_eng,
        mu_complex=complex_energies,
        mu_ligand=MU_LIGAND,
        ligand_concentration=ligand_conc,
        ion_activity=activity,
        T=ROOM_TEMPERATURE_K,
    )

    all_species = []
    for phase, chem_pot in (("bulk", solid_eng), ("aqueous_ion", ion_eng), ("complex", complex_energies)):
        for formula in chem_pot:
            all_species.append(
                Species(
                    formula=formula,
                    metal=metal,
                    thermo=thermo,
                    data=metal_data,
                    phase=phase,
                    activity=0 if phase == "bulk" else activity,
                )
            )

    return thermo, metal_data, all_species


def generate_reactions_for_metal(
    metal: str,
    activity: float,
    ligand_conc: Dict[str, float],
    complex_df,
    grid_size: int,
) -> List[str]:
    thermo, metal_data, all_species = build_species_for_metal(
        metal=metal,
        activity=activity,
        ligand_conc=ligand_conc,
        complex_df=complex_df,
    )

    plotter = GridPlotter(
        pH_range=(-2, 16),
        V_range=(-2, 3),
        data=metal_data,
        grid_size=grid_size,
        save_fig=False,
        dir="",
        filename=None,
    )

    stable_regions = StabilityCalculator(plotter, all_species, thermo, metal_data).compute_stable_regions()
    species_with_area = [sp for sp, mask in stable_regions.items() if np.any(mask)]
    # print(f"Found stable regions for {len(species_with_area)} species: {species_with_area}")
    # print(species_with_area)

    reactions: List[str] = []
    for i, sp_a in enumerate(species_with_area):
        for sp_b in species_with_area[i + 1 :]:
            if not touches(stable_regions[sp_a], stable_regions[sp_b]):
                continue

            # if (sp_b.formula, sp_b.phase) < (sp_a.formula, sp_a.phase):
            #     rxn = Reaction(sp_b, sp_a, thermo, metal_data)
            # else:
            rxn = Reaction(sp_a, sp_b, thermo, metal_data)
            # print(rxn.reactant.formula, rxn.reactant.phase, rxn.product.formula, rxn.product.phase)


            reactions.append(build_reaction_string(rxn))
    # print(set(reactions))
    # 
    return sorted(set(reactions))


def write_latex_table(metal: str, reactions: List[str], output_dir: Path, activity: float, nh3: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{metal}_reactions.tex"

    header = rf"""\begin{{longtable}}{{|p{{0.9cm}}|p{{13.3cm}}|}}
\caption{{Balanced reactions appearing at equilibrium boundaries in the {metal} NH$_3$-H$_2$O Pourbaix diagram ($T={ROOM_TEMPERATURE_K}$ K, $a(M^{{n+}})={activity:.0e}$, [NH$_3$]={nh3} M).}}\\
\hline
\textbf{{No.}} & \textbf{{Balanced reaction}} \\ \hline
\endfirsthead
\hline
\textbf{{No.}} & \textbf{{Balanced reaction}} \\ \hline
\endhead
"""

    rows = [f"{idx} & {reaction} \\\\ \\hline" for idx, reaction in enumerate(reactions, start=1)]
    content = header + "\n".join(rows) + "\n\\end{longtable}\n"

    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate balanced boundary reactions and LaTeX tables for NH3-H2O Pourbaix diagrams at room temperature."
    )
    parser.add_argument("--metals", nargs="*", default=["Ti", "Cu", "Au", "Pd", "Pt", "Ni"])
    parser.add_argument("--activity", type=float, default=1e-4)
    parser.add_argument("--nh3", type=float, default=0.02)
    parser.add_argument("--grid-size", type=int, default=220)
    parser.add_argument("--output-dir", default="data/paper/reactions")
    args = parser.parse_args()

    ligand_conc = {"NH3": args.nh3, "NO2": 0, "Gly": 0, "CN": 0.0}

    data_loader = MetalComplexDataLoader(str(ROOT / "data" / "metal_complex_del_G.json"))
    complex_df = data_loader.load()

    output_dir = ROOT / args.output_dir
    for metal in args.metals:
        reactions = generate_reactions_for_metal(
            metal=metal,
            activity=args.activity,
            ligand_conc=ligand_conc,
            complex_df=complex_df,
            grid_size=args.grid_size,
        )
        write_latex_table(metal, reactions, output_dir, activity=args.activity, nh3=args.nh3)


if __name__ == "__main__":
    main()
