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

ligand_concentration_list = [experiment_concentration, with_CN, no_CN,
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':1e-4},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.05, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.05, 'CN':1e-4}]
ligand_concentration_list = [aqueous_only, experiment_concentration, with_CN, no_CN,
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':1e-4},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.05, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.05, 'CN':1e-4}]
ligand_concentration_list = [aqueous_only, exp, with_CN,
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':1e-4},
                             {'NH3':0, 'NO2':0, 'Gly': 0.1, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.1, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.1, 'CN':1e-4}]

ligand_concentration_list = [{'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':0},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':1e-4},
                             {'NH3':0., 'NO2':0, 'Gly': 0.0, 'CN':1e-2},
                             {'NH3':0.02, 'NO2':0, 'Gly': 0.00, 'CN':0},
                             {'NH3':0.0, 'NO2':0, 'Gly': 0.00, 'CN':0}]
meng_concentration = {'NH3': 1, 'NO2': 0, 'Gly': 0, 'CN': 0}
ligand_concentration_list = [aqueous_only,
                            {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':0},
                            {'NH3':0.02, 'NO2':0, 'Gly': 0.005, 'CN':0.0001}]
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
data_loader.save_to_csv('../data/metal_complex_energies.csv')
# -------------------------------------
# Target Metals and Constants
# -------------------------------------
metal_list = ['Cu','Au','Ni','Pt','Pd','Ti','Co','Cd','Sr','Mg','Mn','Zn','Fe', 'Ag']
metal_list = ['Cu','Au','Ni','Pt','Pd','Ti']
metal_list = ['Pd']

metal_stable_regions = {}
T = 298.15
activity = 1e-4
# diff_list = [-0.1, -0.05, 0, 0.05, 0.075, 0.1, 0.125, 0.15]
# diff_list = [-0.25,-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
def kJmol_to_eV(x):
    return x / 96.485
# -------------------------------------
# Main Loop Over Metals and Ligands
# -------------------------------------
for metal in metal_list:
        for ligand_conc in ligand_concentration_list:
            target_df = df[df['metal'] == metal]
            target_df = df[df['metal'] == metal].copy()
            
            ligand_mu_eV = kJmol_to_eV(target_df["G_ligand (kJ/mol)"])
            # target_df["del_G_eV_adj"] = (
            #     target_df["del_G_eV"]
            #     - ligand_mu_eV * target_df["n_complex"]
            # )
            # metal_complex = target_df.set_index('species')['del_G_eV_adj'].to_dict()
            metal_complex = target_df.set_index('species')['del_G_eV'].to_dict()
            
            # Manually override missing data
            # if metal == 'Cu':
            # #     print(metal_complex)
            #     metal_complex['Cu(Gly)2[2+]'] =-6.77
            # Manually override missing data
            if metal == 'Pd':
                metal_complex['Pd(CN)4[2+]'] = 6.467825128
            # if metal == 'Pt':
            #     metal_complex['Pt(CN)4[2+]'] = 5.646346039


            species_label_dict = target_df.set_index('species')['species_label'].to_dict()

            # Load or compute formation energies
            solid_path = f'data/{metal}_solid_formation_energy.json'
            ion_path = f'data/{metal}_ion_formation_energy.json'

            if os.path.exists(solid_path):
                with open(solid_path, 'r') as f:
                    solid_eng = json.load(f)
            else:
                solid_eng = get_solid_formation_energy(metal, mpr)

            # solid_eng['Ni2O2'] += diff
            if os.path.exists(ion_path):
                with open(ion_path, 'r') as f:
                    ion_eng = json.load(f)
            else:
                ion_eng = get_ion_formation_energy(metal, mpr)

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
            # for phase, chem_pot in [('bulk', solid_eng), ('aqueous_ion', ion_eng)]:
            for phase, chem_pot in [('bulk', solid_eng), ('aqueous_ion', ion_eng), ('complex', metal_complex)]:
                for formula in chem_pot:
                    # if phase == 'bulk':
                    #     activity = 0
                    
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
            filename = None
            # filename =f'Ni-NH3-H2O_T={T}_activity={activity:.0e}_[NH3]={ligand_conc["NH3"]}M_[Gly]={ligand_conc["Gly"]}M_[CN]={ligand_conc["CN"]}_diff={diff}eV.png'
            # filename = f'Pd-NH3-H2O_T={T}_activity={activity:.0e}_[NH3]={ligand_conc["NH3"]}M_[Gly]={ligand_conc["Gly"]}M_[CN]={ligand_conc["CN"]}_Smith1989CriticalConstants.png'
            # filename = f'Pt-NH3-H2O_T={T}_activity={activity:.0e}_[NH3]={ligand_conc["NH3"]}M_[Gly]={ligand_conc["Gly"]}M_[CN]={ligand_conc["CN"]}_Harrington.png'
            # paper_dir = f'/home/x-ntian/pourbaix_paper/Accelerated-Computational-Materials-Discovery-for-Electrochemical-Nutrient-Recovery/Figures/pourbaix_diagrams{metal}'
            dir = f'figures/pourbaix_diagrams/{metal}'
            plotter = GridPlotter(pH_range, V_range, metal_data, grid_size, save_fig=True, dir = dir, filename=filename)

            # Stability and plotting
            stability_calculator = StabilityCalculator(plotter, all_species, thermo, metal_data)
            stable_regions = stability_calculator.compute_stable_regions()
            metal_stable_regions[metal] = stable_regions


            plotter.plot_stable_regions(stable_regions, species_label_dict)
