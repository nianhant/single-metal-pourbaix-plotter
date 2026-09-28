import pandas as pd
import re
import os

stability_constant_path = "../data/log_B.xlsx"

df = pd.read_excel(stability_constant_path)
df['del_G_eV'] = df['del G (eV) Paul'].fillna(df['del G (eV) Azida']).fillna(df['del G (eV) Other']).fillna(df['del G (eV) Other'])
df['log_B'] = df["log_B Paul's Handbook"].fillna(df['log_B Azida Avg'])

df = df[~df['ligand'].str.contains('NH4', na=False)]
df = df[~df['ligand'].str.contains('OH', na=False)]
df = df[~df['ligand'].str.contains('Cl', na=False)]

new_df = df[['ligand', 'metal_ion', 'n_metal', 'n_complex','G_ligand (kJ/mol)','G_metal (kJ/mol)', 'signed_metal_ion', 'del_G_eV','log_B']]
new_df=new_df.dropna(how='any')


def generate_species(row):
    def extract_charge(expression):
        if "[" not in expression or "]" not in expression:
            return 0
        ion = expression[expression.find("[") + 1 : expression.find("]")]
        return int(ion.replace("+", "").replace("-", "")) * (-1 if "-" in ion else 1)

    def format_charge(charge):
        return f"{abs(charge)}{'+' if charge > 0 else '-'}" if charge else ""

    metal = row["signed_metal_ion"].split("[")[0]
    ligand = row["ligand"].split("[")[0]
    n_metal = int(row["n_metal"])
    n_ligand = int(row["n_complex"])
    metal_count = "" if n_metal == 1 else str(n_metal)
    ligand_count = "" if n_ligand == 1 else str(n_ligand)
    charge = extract_charge(row["signed_metal_ion"]) * n_metal + extract_charge(row["ligand"]) * n_ligand
    charge_part = f"[{format_charge(charge)}]" if charge else ""

    return f"{metal}{metal_count}({ligand}){ligand_count}{charge_part}"

new_df['species'] = new_df.apply(generate_species, axis=1)
new_df['metal'] = new_df['signed_metal_ion'].str.split('[').str[0]

if not os.path.exists('../data'):
    os.makedirs('../data')

new_df.to_json('../data/metal_complex_del_G.json', orient='records', indent=4)
print("DataFrame saved to '../data/metal_complex_del_G.json'")
