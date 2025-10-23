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
no_CN = {'NH3': 0.02, 'NO2': 0, 'Gly': 0.005, 'CN': 0}
with_CN = {'NH3': 0.02, 'NO2': 0, 'Gly': 0.005, 'CN': 1e-4}
experiment_concentration = {'NH3':0., 'NO2':0, 'Gly': 0.01, 'CN':0}
ligand_concentration_list = [experiment_concentration]

# -------------------------------------
# Ligand Chemical Potentials
# -------------------------------------
mu_ligand = {
    'NH3': -0.276037,
    'Gly': -3.263014109,
    'CN': 1.786800089,
    'NO2': -0.333729483
}

# -------------------------------------
# Load Metal Complex Energies
# -------------------------------------
del_G_json_path = '../data/metal_complex_del_G.json'
data_loader = MetalComplexDataLoader(del_G_json_path)
df = data_loader.load()
data_loader.save_to_csv('data/metal_complex_energies.csv')
# -------------------------------------
# Target Metals and Constants
# -------------------------------------
metal_list = ['Cu']
activity_list = [1e-2, 1e-3, 1e-4,1e-5, 1e-6,1e-7,1e-8]
activity_list = [1e-1, 1e-2, 1e-3]
activity_list = [ 1e-5]
activity =  1e-6

T_list = [298,308,318,328,338,348,358]
metal_stable_regions = {}
T = 298.15

# diff_list = [-0.1, -0.05, 0, 0.05, 0.075, 0.1, 0.125, 0.15]
# -------------------------------------
# Main Loop Over Metals and Ligands
# -------------------------------------
for metal in metal_list:
    # for diff in diff_list:
    for activity in activity_list:
        for ligand_conc in ligand_concentration_list:
            target_df = df[df['metal'] == metal]
            metal_complex = target_df.set_index('species')['del_G_eV'].to_dict()
            
            # Manually override missing data
            # if metal == 'Cu':
            # #     print(metal_complex)
            #     metal_complex['Cu(Gly)2[2+]'] =-6.77

            species_label_dict = target_df.set_index('species')['species_label'].to_dict()

            # Load or compute formation energies
            solid_path = f'data/{metal}_solid_formation_energy.json'
            ion_path = f'data/{metal}_ion_formation_energy.json'

            if os.path.exists(solid_path):
                with open(solid_path, 'r') as f:
                    solid_eng = json.load(f)
            else:
                solid_eng = get_solid_formation_energy(metal)


            # solid_eng['Cu2O2'] += diff
            if os.path.exists(ion_path):
                with open(ion_path, 'r') as f:
                    ion_eng = json.load(f)
            else:
                ion_eng = get_ion_formation_energy(metal)

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
                        activity=0 if phase == 'bulk' else activity
                    )
                    all_species.append(species)

            # Grid and plotting
            pH_range = (-2, 16)
            V_range = (-2, 3)
            grid_size = 400
            # plotter = GridPlotter(pH_range, V_range, metal_data, grid_size, save_fig=True, dir = 'T_test')
            plotter = GridPlotter(pH_range, V_range, metal_data, grid_size, save_fig=True, dir = 'E_test_glycine', filename=None)#f'figures/E_test_glycine/{diff}.png')

            # Stability and plotting
            stability_calculator = StabilityCalculator(plotter, all_species, thermo, metal_data)
            stable_regions = stability_calculator.compute_stable_regions()
            metal_stable_regions[metal] = stable_regions


            plotter.plot_stable_regions(stable_regions, species_label_dict)
