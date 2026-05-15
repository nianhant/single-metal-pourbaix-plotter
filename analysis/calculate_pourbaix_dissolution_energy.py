import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "methods"
EXAMPLES_DIR = ROOT / "examples"
if str(METHODS_DIR) not in sys.path:
    sys.path.append(str(METHODS_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))

from data import Data
from grid_plotter import GridPlotter
from reaction import Reaction
from species import Species
from thermodynamics import Thermodynamics

from generate_pourbaix import (
    DEFAULT_LIGAND_PRESETS,
    DEFAULT_MU_LIGAND,
    build_complex_energy_dict,
    build_ligand_concentration,
    build_species,
    load_formation_energies,
)
from metal_complex_dataloader import MetalComplexDataLoader

PREFAC_298K = 0.0591


@dataclass(frozen=True)
class DissolutionResult:
    case_label: str
    metal: str
    pH: float
    U_RHE: float
    U_SHE: float
    ligand_concentration: Dict[str, float]
    ion_activity: float
    aqueous_species: str
    aqueous_energy_eV_per_metal: float
    solid_species: str
    solid_energy_eV_per_metal: float
    delta_g_pbx_eV_per_metal: float


class SinglePointGrid:
    def __init__(self, pH: float, U_SHE: float, data):
        self.pH_grid = np.array([[pH]], dtype=float)
        self.V_grid = np.array([[U_SHE]], dtype=float)

        helper_grid = GridPlotter(
            pH_range=(pH, pH),
            V_range=(U_SHE, U_SHE),
            data=data,
            grid_size=1,
            save_fig=False,
            dir="",
            filename="",
        )
        self.ligand_grid_dict = {
            ligand: np.array([[value[0, 0]]], dtype=float)
            for ligand, value in helper_grid.ligand_grid_dict.items()
        }


def energy_at_point(species: Species, thermo, data, grid) -> float:
    return float(Reaction(species, species, thermo, data).calculate_energy_grid(grid)[0, 0])


def build_metal_data(metal: str, activity: float, temperature: float, ligand_concentration: Dict[str, float], use_ligand_correction: bool):
    loader = MetalComplexDataLoader(str(ROOT / "data" / "metal_complex_del_G.json"))
    complex_df = loader.load()
    solid_eng, ion_eng = load_formation_energies(metal)
    metal_complex, _ = build_complex_energy_dict(
        complex_df=complex_df,
        metal=metal,
        use_ligand_correction=use_ligand_correction,
    )

    data = Data(
        metal=metal,
        mu_bulk=solid_eng,
        mu_aqueous_ion=ion_eng,
        mu_complex=metal_complex,
        mu_ligand=DEFAULT_MU_LIGAND,
        ligand_concentration=ligand_concentration,
        ion_activity=activity,
        T=temperature,
    )
    thermo = Thermodynamics(T=temperature)
    species = build_species(metal, thermo, data, solid_eng, ion_eng, metal_complex)
    return data, thermo, species


def calculate_delta_g_pbx(
    case_label: str,
    metal: str,
    pH: float,
    U_RHE: float,
    activity: float,
    temperature: float,
    ligand_concentration: Dict[str, float],
    use_ligand_correction: bool,
) -> DissolutionResult:
    U_SHE = U_RHE - PREFAC_298K * pH
    data, thermo, species = build_metal_data(
        metal=metal,
        activity=activity,
        temperature=temperature,
        ligand_concentration=ligand_concentration,
        use_ligand_correction=use_ligand_correction,
    )
    grid = SinglePointGrid(pH=pH, U_SHE=U_SHE, data=data)

    solid_candidates = [sp for sp in species if sp.phase == "bulk"]
    aqueous_candidates = [sp for sp in species if sp.phase in {"aqueous_ion", "complex"}]

    solid_energies = [(sp, energy_at_point(sp, thermo, data, grid)) for sp in solid_candidates]
    aqueous_energies = [(sp, energy_at_point(sp, thermo, data, grid)) for sp in aqueous_candidates]

    solid_species, solid_energy = min(solid_energies, key=lambda item: item[1])
    aqueous_species, aqueous_energy = min(aqueous_energies, key=lambda item: item[1])

    return DissolutionResult(
        case_label=case_label,
        metal=metal,
        pH=pH,
        U_RHE=U_RHE,
        U_SHE=U_SHE,
        ligand_concentration=dict(ligand_concentration),
        ion_activity=activity,
        aqueous_species=aqueous_species.formula,
        aqueous_energy_eV_per_metal=aqueous_energy,
        solid_species=solid_species.formula,
        solid_energy_eV_per_metal=solid_energy,
        delta_g_pbx_eV_per_metal=aqueous_energy - solid_energy,
    )


def parse_material_formula(formula: str) -> Dict[str, float]:
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not parts:
        raise ValueError(f"Could not parse material formula: {formula}")

    composition = {
        element: float(amount) if amount else 1.0
        for element, amount in parts
    }
    parsed_parts = []
    for element, _ in parts:
        amount = composition[element]
        amount_label = "" if amount == 1.0 else f"{amount:g}"
        parsed_parts.append(f"{element}{amount_label}")
    parsed_formula = "".join(parsed_parts)
    if parsed_formula != formula:
        raise ValueError(f"Could not parse material formula cleanly: {formula}")
    return composition


def normalize_composition(composition: Dict[str, float]) -> Dict[str, float]:
    total = sum(composition.values())
    if total <= 0:
        raise ValueError(f"Composition must have positive stoichiometry: {composition}")
    return {element: amount / total for element, amount in composition.items()}


def build_alloy_result(
    case_label: str,
    alloy: str,
    constituent_results: Dict[Tuple[str, str], DissolutionResult],
) -> DissolutionResult:
    weights = normalize_composition(parse_material_formula(alloy))
    parts = []
    missing = []
    for metal, weight in weights.items():
        result = constituent_results.get((case_label, metal))
        if result is None:
            missing.append(metal)
        else:
            parts.append((weight, result))

    if missing:
        raise ValueError(
            f"Cannot compute {alloy} for case {case_label}; missing constituent results for {', '.join(missing)}."
        )

    reference = parts[0][1]
    ligand_concentration = dict(reference.ligand_concentration)
    aqueous_energy = sum(weight * result.aqueous_energy_eV_per_metal for weight, result in parts)
    solid_energy = sum(weight * result.solid_energy_eV_per_metal for weight, result in parts)
    aqueous_species = "; ".join(f"{result.metal}:{result.aqueous_species}" for _, result in parts)
    solid_species = "; ".join(f"{result.metal}:{result.solid_species}" for _, result in parts)

    return DissolutionResult(
        case_label=case_label,
        metal=alloy,
        pH=reference.pH,
        U_RHE=reference.U_RHE,
        U_SHE=reference.U_SHE,
        ligand_concentration=ligand_concentration,
        ion_activity=reference.ion_activity,
        aqueous_species=f"weighted({aqueous_species})",
        aqueous_energy_eV_per_metal=aqueous_energy,
        solid_species=f"weighted({solid_species})",
        solid_energy_eV_per_metal=solid_energy,
        delta_g_pbx_eV_per_metal=aqueous_energy - solid_energy,
    )


def resolve_ligand_cases(args) -> List[tuple[str, Dict[str, float]]]:
    if args.ligand_presets:
        return [
            (name, dict(DEFAULT_LIGAND_PRESETS[name]))
            for name in args.ligand_presets
        ]
    if args.ligand_preset:
        return [(args.ligand_preset, dict(DEFAULT_LIGAND_PRESETS[args.ligand_preset]))]
    ligand_concentration = build_ligand_concentration(
        nh3=args.nh3,
        no2=args.no2,
        gly=args.gly,
        cn=args.cn,
    )
    label = f"NH3={args.nh3:g}M_Gly={args.gly:g}M_CN={args.cn:g}M"
    return [(label, ligand_concentration)]


def write_csv(results: Iterable[DissolutionResult], path: Path):
    rows = list(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "metal",
        "pH",
        "U_RHE_V",
        "U_SHE_V",
        "ion_activity_M",
        "NH3_M",
        "Gly_M",
        "CN_M",
        "aqueous_species_best",
        "G_aq_best_eV_per_metal",
        "solid_species_ref_best",
        "G_solid_ref_eV_per_metal",
        "delta_G_pbx_best_eV_per_metal",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    "case": result.case_label,
                    "metal": result.metal,
                    "pH": f"{result.pH:.3f}",
                    "U_RHE_V": f"{result.U_RHE:.3f}",
                    "U_SHE_V": f"{result.U_SHE:.3f}",
                    "ion_activity_M": f"{result.ion_activity:.3e}",
                    "NH3_M": f"{result.ligand_concentration.get('NH3', 0.0):.6g}",
                    "Gly_M": f"{result.ligand_concentration.get('Gly', 0.0):.6g}",
                    "CN_M": f"{result.ligand_concentration.get('CN', 0.0):.6g}",
                    "aqueous_species_best": result.aqueous_species,
                    "G_aq_best_eV_per_metal": f"{result.aqueous_energy_eV_per_metal:.6f}",
                    "solid_species_ref_best": result.solid_species,
                    "G_solid_ref_eV_per_metal": f"{result.solid_energy_eV_per_metal:.6f}",
                    "delta_G_pbx_best_eV_per_metal": f"{result.delta_g_pbx_eV_per_metal:.6f}",
                }
            )


def write_markdown(results: Iterable[DissolutionResult], path: Path):
    rows = list(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pourbaix Dissolution Energy",
        "",
        "Definition: Delta G_pbx(U, pH) = G_aq,most stable(U, pH) - G_solid,ref(U, pH).",
        "The solid reference is the most stable solid species for that metal at the same point.",
        "",
        f"Condition: pH {rows[0].pH:.2f}, {rows[0].U_RHE:.2f} V vs RHE ({rows[0].U_SHE:.3f} V vs SHE), activity {rows[0].ion_activity:.0e} M.",
        "",
        "| Case | Metal | NH3 (M) | Gly (M) | CN (M) | Best aq species | G_aq best (eV/metal) | Best solid ref | G_solid ref (eV/metal) | Delta G_pbx (eV/metal) |",
        "|---|---|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for result in rows:
        lines.append(
            f"| {result.case_label} | {result.metal} | "
            f"{result.ligand_concentration.get('NH3', 0.0):g} | "
            f"{result.ligand_concentration.get('Gly', 0.0):g} | "
            f"{result.ligand_concentration.get('CN', 0.0):g} | "
            f"{result.aqueous_species} | {result.aqueous_energy_eV_per_metal:.3f} | "
            f"{result.solid_species} | {result.solid_energy_eV_per_metal:.3f} | "
            f"{result.delta_g_pbx_eV_per_metal:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Calculate Pourbaix dissolution energy at one experimental point.")
    parser.add_argument("--metals", nargs="+", default=["Au", "Pd", "Ni", "Cu"])
    parser.add_argument("--alloys", nargs="*", default=["AuPd", "TiCu", "TiNi"])
    parser.add_argument("--ph", type=float, default=12.0)
    parser.add_argument("--u-rhe", type=float, default=2.0)
    parser.add_argument("--activity", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--ligand-preset", choices=sorted(DEFAULT_LIGAND_PRESETS), default=None)
    parser.add_argument("--ligand-presets", nargs="+", choices=sorted(DEFAULT_LIGAND_PRESETS), default=None)
    parser.add_argument("--nh3", type=float, default=0.0)
    parser.add_argument("--no2", type=float, default=0.0)
    parser.add_argument("--gly", type=float, default=0.1)
    parser.add_argument("--cn", type=float, default=0.0)
    parser.add_argument("--use-ligand-correction", action="store_true")
    parser.add_argument("--no-output-files", action="store_true")
    parser.add_argument("--output-csv", default="analysis/pourbaix_dissolution_energy.csv")
    parser.add_argument("--output-md", default="analysis/pourbaix_dissolution_energy.md")
    return parser


def main():
    args = build_parser().parse_args()
    ligand_cases = resolve_ligand_cases(args)
    alloy_compositions = {
        alloy: parse_material_formula(alloy)
        for alloy in args.alloys
    }
    metals = list(dict.fromkeys(
        list(args.metals)
        + [
            metal
            for composition in alloy_compositions.values()
            for metal in composition
        ]
    ))
    results: List[DissolutionResult] = [
        calculate_delta_g_pbx(
            case_label=case_label,
            metal=metal,
            pH=args.ph,
            U_RHE=args.u_rhe,
            activity=args.activity,
            temperature=args.temperature,
            ligand_concentration=ligand_concentration,
            use_ligand_correction=args.use_ligand_correction,
        )
        for case_label, ligand_concentration in ligand_cases
        for metal in metals
    ]
    constituent_results = {
        (result.case_label, result.metal): result
        for result in results
    }
    alloy_results = [
        build_alloy_result(
            case_label=case_label,
            alloy=alloy,
            constituent_results=constituent_results,
        )
        for case_label, _ in ligand_cases
        for alloy in args.alloys
    ]
    results.extend(alloy_results)

    if not args.no_output_files:
        write_csv(results, ROOT / args.output_csv)
        write_markdown(results, ROOT / args.output_md)

    for result in results:
        print(
            f"{result.case_label} {result.metal}: "
            f"Delta G_pbx = {result.delta_g_pbx_eV_per_metal:.3f} eV/metal "
            f"({result.aqueous_species} - {result.solid_species})"
        )


if __name__ == "__main__":
    main()
