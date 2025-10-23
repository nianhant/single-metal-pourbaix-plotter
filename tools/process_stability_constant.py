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
    signed_metal = row['signed_metal_ion'].split('[')
    ligand_part = row['ligand'].split('[')[0]
    
    n_metal = int(row['n_metal'])  # Ensure n_metal is an integer
    n_metal_part = str(n_metal) if n_metal != 1 else ''
    
    if len(signed_metal) > 1:
        ion_part = signed_metal[1].strip(']')  # Remove closing bracket
        # Use regular expression to extract the number and charge
        match = re.match(r'(\d+)([+-])', ion_part)
        if match:
            number = int(match.group(1)) * n_metal  # Multiply the number by n_metal
            charge_sign = match.group(2)  # Extract the sign
            charge_multiplier = 1 if charge_sign == '+' else -1  # Apply the charge as multiplier
            
            result = number * charge_multiplier  # Multiply the number by the charge
 
            signed_metal_ion_part = f'[{result}{charge_sign}]'
 
        else:
            signed_metal_ion_part = ''  # Default case if not matched
    else:
        signed_metal_ion_part = ''
    
    species = signed_metal[0] + n_metal_part + '(' + ligand_part + ')' + str(int(row['n_complex'])) + signed_metal_ion_part
    
    return species

new_df['species'] = new_df.apply(generate_species, axis=1)
new_df['metal'] = new_df['signed_metal_ion'].str.split('[').str[0]

if not os.path.exists('../data'):
    os.makedirs('../data')

new_df.to_json('../data/metal_complex_del_G.json', orient='records', indent=4)
print("DataFrame saved to '../data/metal_complex_del_G.json'")
