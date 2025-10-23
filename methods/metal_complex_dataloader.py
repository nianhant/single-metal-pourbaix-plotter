import os
import json
import pandas as pd
from pymatgen_api import get_ion_formation_energy, get_solid_formation_energy

class MetalComplexDataLoader:
    def __init__(self, json_path):
        self.json_path = json_path
        self.df = None

    def load(self):
        """Load JSON and process species label"""
        self.df = pd.read_json(self.json_path)
        self.df["species_label"] = self.df.apply(self.create_species, axis=1)
        return self.df

    def save_to_csv(self, output_path):
        """Save processed DataFrame to CSV"""
        if self.df is None:
            raise ValueError("Data has not been loaded. Call `.load()` first.")
        self.df.to_csv(output_path, index=False)

    @staticmethod
    def _extract_ion_number(expression):
        """Extract formal charge from an ion string like 'Fe[3+]'"""
        if "[" in expression and "]" in expression:
            ion = expression[expression.find("[") + 1 : expression.find("]")]
            return int(ion.replace("+", "").replace("-", "")) * (-1 if "-" in ion else 1)
        return 0

    @staticmethod
    def _format_charge(charge):
        if charge == 0:
            return ""
        sign = "^+" if charge > 0 else "^-"
        return f"{abs(charge)}{sign}"

    def _create_species(self, row):
        metal_charge = self._extract_ion_number(row["signed_metal_ion"]) * row["n_metal"]
        ligand_charge = self._extract_ion_number(row["ligand"]) * row["n_complex"]
        total_charge = metal_charge + ligand_charge

        metal_ion = row["signed_metal_ion"].split("[")[0]
        ligand = row["ligand"].split("[")[0]
        n_metal = str(row["n_metal"]).replace("1", "")
        n_ligand = str(row["n_complex"]).replace("1", "")

        if total_charge == 0:
            charge_superscript = ""
        else:
            charge_superscript = "^"

        species = f"[{metal_ion}{n_metal}({ligand}){n_ligand}]{charge_superscript}{self._format_charge(total_charge)}"
        species = (
            species.replace("_1", "")
            .replace("^1", "")
            .replace("+1", "+")
            .replace("-1", "-")
        )
        return species

    def create_species(self, row):
        """Formats the species label based on charge balance."""
        def extract_ion_number(expression):
            if "[" in expression and "]" in expression:
                ion = expression[expression.find("[") + 1 : expression.find("]")]
                return int(ion.replace("+", "").replace("-", "")) * (-1 if "-" in ion else 1)
            return 0

        def format_charge(charge):
            return f"{abs(charge)}{'+' if charge > 0 else '-'}" if charge else ""

        metal_charge = extract_ion_number(row["signed_metal_ion"]) * row["n_metal"]
        ligand_charge = extract_ion_number(row["ligand"]) * row["n_complex"]
        total_charge = metal_charge + ligand_charge
        total_charge_str = format_charge(total_charge)

        species = f"[{row['signed_metal_ion'].split('[')[0]}{row['n_metal']}({row['ligand'].split('[')[0]}){row['n_complex']}]{total_charge_str}"
        species = f"{row['signed_metal_ion'].split('[')[0]}{row['n_metal']}({row['ligand'].split('[')[0]}){row['n_complex']}[{total_charge_str}]"

        return species.replace("_1", "").replace("^1", "").replace("+1", "+").replace("-1", "-").replace("[]","")
 
