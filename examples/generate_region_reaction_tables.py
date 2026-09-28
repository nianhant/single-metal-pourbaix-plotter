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
from metal_complex_dataloader import MetalComplexDataLoader
from reaction import Reaction
from species import Species
from stability_calculator import StabilityCalculator
from thermodynamics import Thermodynamics
from grid_plotter import GridPlotter


MU_LIGAND = {
    "NH3": -0.276037,
    "Gly": -3.263014109,
    "CN": 1.786800089,
    "NO2": -0.333729483,
}


def kJmol_to_eV(value):
    return value / 96.485


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
    if isinstance(value, float):
        return f"{value:.8g} "
    return f"{value} "


def build_reaction_string(reaction: Reaction) -> str:
    mb = reaction.compute_mass_balance()

    reactants: List[Tuple[float, str]] = []
    products: List[Tuple[float, str]] = []

    reactants.append((mb["fraction"], species_latex(reaction.reactant.formula, reaction.reactant.phase)))
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
        products.append((mb["n_charge"], r"\ce{e-}"))
    elif mb["n_charge"] < 0:
        reactants.append((-mb["n_charge"], r"\ce{e-}"))

    for ligand, count in mb["n_L"].items():
        ligand_tex = fr"\ce{{{ligand}}}"
        if count > 0:
            products.append((count, ligand_tex))
        elif count < 0:
            reactants.append((-count, ligand_tex))

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


def is_pure_metal_solid(species: Species) -> bool:
    return species.phase == "bulk" and species.formula == species.metal


def reaction_with_preferred_orientation(sp_a: Species, sp_b: Species, thermo, metal_data) -> Reaction:
    if is_pure_metal_solid(sp_b) and not is_pure_metal_solid(sp_a):
        return Reaction(sp_b, sp_a, thermo, metal_data)
    return Reaction(sp_a, sp_b, thermo, metal_data)


def build_species_for_metal(
    metal: str,
    activity: float,
    temperature: float,
    ligand_conc: Dict[str, float],
    complex_df,
    adjust_complex_energies: bool,
):
    thermo = Thermodynamics(T=temperature)

    with open(ROOT / "examples" / "data" / f"{metal}_solid_formation_energy.json", "r", encoding="utf-8") as f:
        solid_eng = json.load(f)
    with open(ROOT / "examples" / "data" / f"{metal}_ion_formation_energy.json", "r", encoding="utf-8") as f:
        ion_eng = json.load(f)

    target_df = complex_df[complex_df["metal"] == metal]
    if adjust_complex_energies:
        target_df = target_df.copy()
        target_df["del_G_eV_adjusted"] = (
            target_df["del_G_eV"]
            + kJmol_to_eV(target_df["G_ligand (kJ/mol)"]) * target_df["n_complex"]
        )
        complex_energies = target_df.set_index("species")["del_G_eV_adjusted"].to_dict()
    else:
        complex_energies = target_df.set_index("species")["del_G_eV"].to_dict()

    if metal == "Pd":
        complex_energies.setdefault("Pd(CN)4[2-]", 6.467825128)
    if metal == "Pt":
        complex_energies.setdefault("Pt(CN)4[2-]", 5.646346039)

    metal_data = Data(
        metal=metal,
        mu_bulk=solid_eng,
        mu_aqueous_ion=ion_eng,
        mu_complex=complex_energies,
        mu_ligand=MU_LIGAND,
        ligand_concentration=ligand_conc,
        ion_activity=activity,
        T=temperature,
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


def compute_stable_regions_for_metal(
    metal: str,
    activity: float,
    temperature: float,
    ligand_conc: Dict[str, float],
    complex_df,
    grid_size: int,
    adjust_complex_energies: bool,
):
    thermo, metal_data, all_species = build_species_for_metal(
        metal=metal,
        activity=activity,
        temperature=temperature,
        ligand_conc=ligand_conc,
        complex_df=complex_df,
        adjust_complex_energies=adjust_complex_energies,
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
    return thermo, metal_data, stable_regions


def species_with_visible_regions(stable_regions):
    return [sp for sp, mask in stable_regions.items() if np.any(mask)]


def generate_reaction_table_for_metal(
    metal: str,
    activity: float,
    temperature: float,
    ligand_conc: Dict[str, float],
    complex_df,
    grid_size: int,
    adjust_complex_energies: bool,
):
    thermo, metal_data, stable_regions = compute_stable_regions_for_metal(
        metal=metal,
        activity=activity,
        temperature=temperature,
        ligand_conc=ligand_conc,
        complex_df=complex_df,
        grid_size=grid_size,
        adjust_complex_energies=adjust_complex_energies,
    )
    species_with_area = species_with_visible_regions(stable_regions)

    reactions = []
    for i, sp_a in enumerate(species_with_area):
        for sp_b in species_with_area[i + 1 :]:
            mask_a = stable_regions[sp_a]
            mask_b = stable_regions[sp_b]
            if not touches(mask_a, mask_b):
                continue

            forward = Reaction(sp_a, sp_b, thermo, metal_data)
            backward = Reaction(sp_b, sp_a, thermo, metal_data)
            if is_pure_metal_solid(sp_b) and not is_pure_metal_solid(sp_a):
                sp_left, sp_right, rxn = sp_b, sp_a, backward
            elif is_pure_metal_solid(sp_a) and not is_pure_metal_solid(sp_b):
                sp_left, sp_right, rxn = sp_a, sp_b, forward
            elif (sp_b.formula, sp_b.phase) < (sp_a.formula, sp_a.phase):
                sp_left, sp_right, rxn = sp_b, sp_a, backward
            else:
                sp_left, sp_right, rxn = sp_a, sp_b, forward

            reactions.append(
                {
                    "region_a": species_latex(sp_left.formula, sp_left.phase),
                    "region_b": species_latex(sp_right.formula, sp_right.phase),
                    "reaction": build_reaction_string(rxn),
                }
            )

    reactions.sort(key=lambda row: (row["region_a"], row["region_b"]))
    return reactions


def generate_reactions_for_metal(
    metal: str,
    activity: float,
    temperature: float,
    ligand_conc: Dict[str, float],
    complex_df,
    grid_size: int,
    adjust_complex_energies: bool,
) -> List[str]:
    thermo, metal_data, stable_regions = compute_stable_regions_for_metal(
        metal=metal,
        activity=activity,
        temperature=temperature,
        ligand_conc=ligand_conc,
        complex_df=complex_df,
        grid_size=grid_size,
        adjust_complex_energies=adjust_complex_energies,
    )
    species_with_area = species_with_visible_regions(stable_regions)

    reactions: List[str] = []
    for i, sp_a in enumerate(species_with_area):
        for sp_b in species_with_area[i + 1 :]:
            if not touches(stable_regions[sp_a], stable_regions[sp_b]):
                continue
            reactions.append(
                build_reaction_string(reaction_with_preferred_orientation(sp_a, sp_b, thermo, metal_data))
            )

    return sorted(set(reactions))


def write_region_latex_table(metal: str, reactions: List[dict], output_dir: Path, caption_suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{metal}_region_reactions.tex"

    header = rf"""\begin{{longtable}}{{|p{{2.6cm}}|p{{2.6cm}}|p{{8.8cm}}|}}
\caption{{Balanced inter-region reactions for {metal} NH$_3$-H$_2$O Pourbaix diagram ({caption_suffix}).}}\\
\hline
\textbf{{Region A}} & \textbf{{Region B}} & \textbf{{Balanced reaction}} \\ \hline
\endfirsthead
\hline
\textbf{{Region A}} & \textbf{{Region B}} & \textbf{{Balanced reaction}} \\ \hline
\endhead
"""

    rows = []
    for row in reactions:
        rows.append(f"{row['region_a']} & {row['region_b']} & {row['reaction']} \\\\ \\hline")
    if not rows:
        rows.append(
            r"\multicolumn{3}{|l|}{No adjacent stable-region boundaries were found for these conditions.} \\ \hline"
        )

    content = header + "\n".join(rows) + "\n\\end{longtable}\n"
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def write_reaction_latex_table(metal: str, reactions: List[str], output_dir: Path, caption_suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{metal}_reactions.tex"

    header = rf"""\begin{{longtable}}{{|p{{0.9cm}}|p{{13.3cm}}|}}
\caption{{Balanced reactions appearing at equilibrium boundaries in the {metal} NH$_3$-H$_2$O Pourbaix diagram ({caption_suffix}).}}\\
\hline
\textbf{{No.}} & \textbf{{Balanced reaction}} \\ \hline
\endfirsthead
\hline
\textbf{{No.}} & \textbf{{Balanced reaction}} \\ \hline
\endhead
"""

    rows = [f"{idx} & {reaction} \\\\ \\hline" for idx, reaction in enumerate(reactions, start=1)]
    if not rows:
        rows.append(
            r"\multicolumn{2}{|l|}{No adjacent stable-region boundaries were found for these conditions.} \\ \hline"
        )
    content = header + "\n".join(rows) + "\n\\end{longtable}\n"
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced boundary reaction LaTeX tables for NH3-Gly-H2O Pourbaix diagrams."
    )
    parser.add_argument("--metals", nargs="*", default=["Ti", "Cu", "Au", "Pd", "Pt", "Ni"])
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--activity", type=float, default=1e-4)
    parser.add_argument("--nh3", type=float, default=0.02)
    parser.add_argument("--gly", type=float, default=0.1)
    parser.add_argument("--cn", type=float, default=0.0)
    parser.add_argument("--grid-size", type=int, default=220)
    parser.add_argument("--output-dir", default="data/paper/reactions")
    parser.add_argument(
        "--include-regions",
        action="store_true",
        help="Write three-column region-pair tables instead of compact reaction-only tables.",
    )
    parser.add_argument(
        "--raw-complex-energies",
        action="store_true",
        help="Deprecated compatibility flag; raw complex energies are now the default.",
    )
    parser.add_argument(
        "--adjust-complex-energies",
        action="store_true",
        help="Use the ligand-adjusted complex energies used by the Pourbaix diagram script.",
    )
    args = parser.parse_args()

    ligand_conc = {"NH3": args.nh3, "NO2": 0.0, "Gly": args.gly, "CN": args.cn}
    caption_suffix = (
        f"$T={args.temperature}$ K, $a(M^{{n+}})={args.activity:.0e}$, "
        f"[NH$_3$]={args.nh3} M, [Gly]={args.gly} M, [CN]={args.cn} M"
    )
    adjust_complex_energies = args.adjust_complex_energies and not args.raw_complex_energies

    data_loader = MetalComplexDataLoader(str(ROOT / "data" / "metal_complex_del_G.json"))
    complex_df = data_loader.load()

    out_dir = ROOT / args.output_dir
    for metal in args.metals:
        if args.include_regions:
            reactions = generate_reaction_table_for_metal(
                metal=metal,
                activity=args.activity,
                temperature=args.temperature,
                ligand_conc=ligand_conc,
                complex_df=complex_df,
                grid_size=args.grid_size,
                adjust_complex_energies=adjust_complex_energies,
            )
            write_region_latex_table(metal, reactions, out_dir, caption_suffix)
        else:
            reactions = generate_reactions_for_metal(
                metal=metal,
                activity=args.activity,
                temperature=args.temperature,
                ligand_conc=ligand_conc,
                complex_df=complex_df,
                grid_size=args.grid_size,
                adjust_complex_energies=adjust_complex_energies,
            )
            write_reaction_latex_table(metal, reactions, out_dir, caption_suffix)


if __name__ == "__main__":
    main()
