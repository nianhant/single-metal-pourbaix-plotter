import pandas as pd
from pathlib import Path


df = pd.read_excel('../data/log_B.xlsx')

df['del_G_eV'] = df['del G (eV) Paul'].fillna(df['del G (eV) Azida']).fillna(df['del G (eV) Other'])

df = df[~df['ligand'].str.contains('NH4', na=False)]
df = df[~df['ligand'].str.contains('OH', na=False)]
df = df[~df['ligand'].str.contains('Cl', na=False)]

new_df = df[['ligand', 'metal_ion', 'n_metal', 'n_complex', 'signed_metal_ion', 'del_G_eV','reference']]
new_df=new_df.dropna(how='any')


def extract_ion_number(expression):
    if "[" in expression and "]" in expression:
        ion = expression[expression.find("[") + 1 : expression.find("]")]
        return int(ion.replace("+", "").replace("-", "")) * (-1 if "-" in ion else 1)
    return 0

def format_charge(charge):
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    return f"{abs(charge)}{sign}"

def create_species(row):
    # Extract charges
    metal_charge = extract_ion_number(row["signed_metal_ion"]) * row["n_metal"]
    ligand_charge = extract_ion_number(row["ligand"]) * row["n_complex"]
    total_charge = metal_charge + ligand_charge

    # Format total charge
    total_charge_str = format_charge(int(total_charge))

    # Format Species column
    metal_ion = row['signed_metal_ion'].split('[')[0]
    ligand = row['ligand'].split('[')[0]
    n_metal = str(int(row['n_metal'])).replace("1","")
    n_ligand = str(int(row['n_complex'])).replace("1","")
    
    if total_charge == 0:
        total_charge_super_script = ""
    else:
        total_charge_super_script = "^"
    
    species = f"[{metal_ion}{n_metal}({ligand}){n_ligand}]{total_charge_super_script}{total_charge_str}"
    return species.replace("_1","").replace("^1","").replace("+1","+").replace("-1","-")

new_df["Species"] = new_df.apply(create_species, axis=1)
new_df["Metal ion"] = new_df["signed_metal_ion"].str.replace("[", "^").str.replace("]","")

def generate_ligand_complex_latex_table(df, ligand, outdir):
    combined_data = []
    ligand_data = df[df['ligand'] == ligand]
    for _, row in ligand_data.iterrows():
        combined_data.append({
            'Species': row['Species'],
            'Metal ion': row['Metal ion'],
            'del_G_eV': row['del_G_eV'],
            'Reference': row['reference']
        })

    if ligand == 'Gly[1-]':
        ligand_text = 'Gly-'
    if ligand == 'CN[1-]':
        ligand_text = 'CN-'
    if ligand == 'NH3':
        ligand_text = 'NH3'
    latex_table = fr"""\clearpage
\begin{{longtable}}{{|p{{4cm}}|p{{4cm}}|p{{3cm}}|p{{3cm}}|}}
\caption{{Formation energies of species for \ce{{{ligand_text}}} complexes.}} 
\label{{tab:{ligand}_complex_energies}}
\\
\hline
\textbf{{Species}} & \textbf{{Metal ion}} & \textbf{{\( \Delta G^\circ_{{298}} \) (eV)}} & \textbf{{Reference}} \\ \hline
\endfirsthead
\caption*{{Table \thetable\ continued from previous pages.}} \\
\hline
\textbf{{Species}} & \textbf{{Metal ion}} & \textbf{{\( \Delta G^\circ_{{298}} \) (eV)}} & \textbf{{Reference}} \\ \hline
\endhead
\hline
\endfoot
\hline
\endlastfoot
"""
    # Add rows to the table
    for i, entry in enumerate(combined_data):
        latex_table += f"\ce{{{entry['Species']}}} & \ce{{{entry['Metal ion']}}} & {entry['del_G_eV']:.3f} & \\textnormal{{\\citenum{{{entry['Reference']}}}}}"
        if i < len(combined_data) - 1:
            latex_table += " \\\\ \\hline\n"

    latex_table += r"\end{longtable}"

    # output_path = Path(f'../data/paper/ligand/{ligand}_complex_energy_table.tex')
    output_path = Path(outdir) / f"{ligand}_complex_energy_table.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as tex_file:
        tex_file.write(latex_table)
    print(f"LaTeX table saved to {output_path}")

import json 

def format_species(species):
#     print(species)
    species_text = species.split('(aq)')[0]
    total_charge = extract_ion_number(species_text)
    total_charge_str = format_charge(total_charge)
    if total_charge == 0:
        total_charge_super_script = ""
    else:
        total_charge_super_script = "^"
    species_text = f"{species_text.split('[')[0]}{total_charge_super_script}{total_charge_str}"
    return species_text.replace("_1","").replace("^1","").replace("+1","+").replace("-1","-")

def generate_bulk_metal_latex_table(metal, outdir):
    with open(f'../examples/data/{metal}_ion_formation_energy.json', 'r') as ion_file:
        ion_data = json.load(ion_file)
    with open(f'../examples/data/{metal}_solid_formation_energy.json', 'r') as solid_file:
        solid_data = json.load(solid_file)
        
    combined_data = []
    for species, energy in ion_data.items():
        species_name = format_species(species)
        combined_data.append({'Species': species_name, 'State': 'Aqueous ion', 'Energy': energy})
    for species, energy in solid_data.items():
        species_name = format_species(species)
        combined_data.append({'Species': species_name, 'State': 'Solid', 'Energy': energy})
    latex_table = fr"""\clearpage
\begin{{longtable}}{{|p{{4cm}}|p{{3cm}}|p{{3cm}}|}}
\caption{{Formation energies of {metal} species queried from Materials Project\cite{{Jain2013TheInnovation}}.}} 
\label{{tab:bulk_{metal}_energies}}
\\
\hline
\textbf{{Species}}  & \textbf{{State}} & \textbf{{\( \Delta G\) (eV)}} \\ \hline
\endfirsthead
\caption*{{Table \thetable\ continued from previous pages.}} \\
\hline
\textbf{{Species}}  & \textbf{{State}} & \textbf{{\( \Delta G\) (eV)}} \\ \hline
\endhead
\hline
\endfoot
\hline
\endlastfoot
"""
    
    for i, entry in enumerate(combined_data):
        latex_table += f"\ce{{{entry['Species']}}} & {entry['State']} & {entry['Energy']:.3f}"
        if i < len(combined_data) - 1:
            latex_table += " \\\\ \\hline\n"
    latex_table += r"\end{longtable}"
    # output_path = Path(f'../data/paper/metal/{metal}_energy_table.tex')
    output_path = Path(outdir) / f"{metal}_energy_table.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as tex_file:
        tex_file.write(latex_table)
    print(f"LaTeX table saved to {output_path}")
    
ligand_outdir = '/home/x-ntian/pourbaix_paper/Accelerated-Computational-Materials-Discovery-for-Electrochemical-Nutrient-Recovery/data/ligand'

generate_ligand_complex_latex_table(new_df, 'NH3',ligand_outdir)
generate_ligand_complex_latex_table(new_df, 'Gly[1-]',ligand_outdir)
generate_ligand_complex_latex_table(new_df, 'CN[1-]',ligand_outdir)

metal_list = ['Ti', 'Cu','Au','Pd','Pt','Ni']
metal_outdir = '/home/x-ntian/pourbaix_paper/Accelerated-Computational-Materials-Discovery-for-Electrochemical-Nutrient-Recovery/data/metal'

for m in metal_list:
    
    generate_bulk_metal_latex_table(m, metal_outdir)


