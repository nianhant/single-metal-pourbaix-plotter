import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../methods')))


import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen_api import get_ion_formation_energy, get_solid_formation_energy
from data import Data
from species import Species
from grid_plotter import GridPlotter
from thermodynamics import Thermodynamics
from stability_calculator import StabilityCalculator
from metal_complex_dataloader import MetalComplexDataLoader
from mp_api.client import MPRester



mpr_key = "hhsFnwPlqjxA77yv1zKSYGbynYuPJpR6"
mpr = MPRester(mpr_key)
# -------------------------------------
# Plotting Style
# -------------------------------------
def set_publication_style():
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 27,
        'axes.titlesize': 16,
        'xtick.labelsize': 27,
        'ytick.labelsize': 27,
        'legend.fontsize': 14,
        'figure.titlesize': 27,
        'savefig.dpi': 300,
    })

set_publication_style()

# -------------------------------------
# Ligand Concentration Setup
# -------------------------------------
aqueous_only = {'NH3': 0, 'NO2': 0, 'Gly': 0, 'CN': 0}
no_CN = {'NH3': 0.02, 'NO2': 0, 'Gly': 0.1, 'CN': 0}
with_CN = {'NH3': 0.02, 'NO2': 0, 'Gly': 0.1, 'CN': 1e-4}
experiment_concentration = {'NH3':0., 'NO2':0, 'Gly': 0.1, 'CN':0}
exp = {'NH3':0.02, 'NO2':0, 'Gly': 0.05, 'CN':0}

ligand_concentration_list = [{'NH3': 0, 'NO2': 0, 'Gly': 0, 'CN': 0},
                             {'NH3': 0.02, 'NO2': 0, 'Gly': 0.005, 'CN': 1e-4},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':0},
                             {'NH3': 0.02, 'NO2': 0, 'Gly': 0.05, 'CN': 1e-4},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.05, 'CN':0},]

mu_ligand = {
    'NH3': -0.276037,
    'Gly': -3.263014109,
    'CN': 1.786800089,
    'NO2': -0.333729483
}


del_G_json_path = '../data/metal_complex_del_G.json'
data_loader = MetalComplexDataLoader(del_G_json_path)
df = data_loader.load()
data_loader.save_to_csv('../data/metal_complex_energies.csv')

metal_list = ['Ni']

metal_stable_regions = {}
T = 298.15
activity = 1e-4

def kJmol_to_eV(x):
    return x / 96.485

for metal in metal_list:
        for ligand_conc in ligand_concentration_list:
            target_df = df[df['metal'] == metal]
            target_df = df[df['metal'] == metal].copy()
            
            ligand_mu_eV = kJmol_to_eV(target_df["G_ligand (kJ/mol)"])
            metal_complex = target_df.set_index('species')['del_G_eV'].to_dict()
            
            if metal == 'Pd':
                metal_complex['Pd(CN)4[2+]'] = 6.467825128
            if metal == 'Pt':
                metal_complex['Pt(CN)4[2+]'] = 5.646346039


            species_label_dict = target_df.set_index('species')['species_label'].to_dict()
            solid_eng = get_solid_formation_energy(metal, mpr, stable_only=True, save_json=False)
            ion_eng = get_ion_formation_energy(metal, mpr, save_json=False)

            if 'FeOH[2+]' in ion_eng:
                ion_eng.pop('FeOH[2+]')

            # Create Data object
            metal_data = Data(
                metal=metal,
                mu_bulk=solid_eng,
                mu_aqueous_ion=ion_eng,
                mu_complex=metal_complex,
                mu_ligand=mu_ligand,
                ligand_concentration=ligand_conc,
                ion_activity=activity,
                T = T
            )

            thermo = Thermodynamics(T = T)

            # Combine species
            all_species = []
            for phase, chem_pot in [('bulk', solid_eng), ('aqueous_ion', ion_eng), ('complex', metal_complex)]:
                for formula in chem_pot:
                    species = Species(
                        formula=formula,
                        metal=metal,
                        thermo=thermo,
                        data=metal_data,
                        phase=phase,
                        activity= activity
                    )
                    if phase == 'complex':
                        print(f"Complex {formula} has activity {species.activity}")
                    all_species.append(species)

            # Grid and plotting
            pH_range = (-2, 16)
            V_range = (-2, 3)
            grid_size = 1000
            filename = 'pdf'
            # filename = None
            
            dir = f'/global/homes/n/nianhant/data/stability_paper/stability_manuscript/Figures/TOC_figures/{metal}'
            plotter = GridPlotter(pH_range, V_range, metal_data, grid_size, save_fig=True, dir = dir, filename=filename)

            # Stability and plotting
            stability_calculator = StabilityCalculator(plotter, all_species, thermo, metal_data)
            stable_regions = stability_calculator.compute_stable_regions()
            metal_stable_regions[metal] = stable_regions


            plotter.plot_stable_regions(stable_regions, species_label_dict, rxn_box=True)
