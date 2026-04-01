import copy
import numpy as np 
import re

class Reaction:
    def __init__(self, reactant, product, thermo, data):
        self.reactant = reactant
        self.product = product
        self.thermo = thermo
        self.data = data

    def calculate_energy_grid(self,grid):
        
        composition = copy.deepcopy(self.reactant.composition)
        # print(composition)

        #  e = e0 + 0.0591 log10(conc) - nO mu_H2O
        #     + (nH - 2nO) pH + phi (-nH + 2nO + q)

        n_H2O = composition['O']
        n_H = composition['H'] - 2*n_H2O 
        n_eu = composition['charge'] - n_H

        n_metal = composition[self.reactant.metal]
        n_ligand = 0
        ligand = None
        ligand_standard_energy = 0
        p_ligand_grid = 0

        coeff = self.thermo.kB*self.thermo.T*np.log(10)

        ligand_coeff = 1

        if self.reactant.ligand:
            ligand = self.reactant.ligand
            n_ligand = composition[ligand]
            ligand_standard_energy = n_ligand  * self.data.mu_ligand[ligand]
            a = max(self.data.ligand_concentration[ligand], 1e-30)
            # ligand_coeff = self.thermo.kB*self.thermo.T*np.log(a)

            p_ligand_grid = grid.ligand_grid_dict[ligand]
            # print(p_ligand_grid)
        
        energy_grid = ( self.reactant.mu 
                       + n_H * coeff * grid.pH_grid  
                       - n_eu * grid.V_grid
                       - n_H2O*self.thermo.mu_H2O 
                       - ligand_standard_energy)
        
        energy_grid += n_ligand * coeff * p_ligand_grid
        energy_grid /= n_metal
        return energy_grid

        
    def compute_mass_balance(self):
        reactant = self.reactant 
        product = self.product
        # return a dictionary of mass balances
        rea_composition = copy.deepcopy(self.reactant.composition)
        prod_composition = copy.deepcopy(self.product.composition)

        mass_balance_comp = {'fraction':1,
                             'n_H2O':0, # RHS of equation
                             'n_charge':0, # LHS of equation
                             'n_L':{},
                             'n_H':0
                            } # LHS of equation
        ############## Balance metal elements ##############
        fraction = prod_composition[product.metal]/rea_composition[reactant.metal]
        for key in rea_composition:
            rea_composition[key] *= fraction

        # ############## Balance ligand ##############
        if not reactant.ligand and not product.ligand:
            n_L = {}
        elif reactant.ligand is None: # product has ligand
            # print('product has ligand')
            # print('prod_composition', prod_composition)
            n_L = {product.ligand: -prod_composition[product.ligand]}
        elif product.ligand is None: # rea has ligand
            n_L = {reactant.ligand: rea_composition[reactant.ligand]}

        else:
            if reactant.ligand == product.ligand: # have same ligand
                n_L = {reactant.ligand: rea_composition[reactant.ligand]- prod_composition[reactant.ligand]}
            else: # have different ligand
                n_L = {reactant.ligand: rea_composition[reactant.ligand],
                       product.ligand: - prod_composition[product.ligand]}

        ############## Balance O ##############
        n_H2O = rea_composition['O'] - prod_composition['O'] 
        ############## Balance H protons ##############
        n_H = 2*n_H2O + prod_composition['H'] - rea_composition['H']
        ############## Balance charge ##############
        n_charge = prod_composition['charge'] - rea_composition['charge']- n_H
        mass_balance_comp = {'fraction': fraction,
                             'n_H2O':n_H2O,
                             'n_charge':n_charge,
                             'n_H':n_H,
                             'n_L':n_L}
        return mass_balance_comp
    
    def formulate_coefficients(self):
        """
        Calculate the chemical potential difference for a single reaction.
        """
        data = self.data
        reactant = self.reactant
        product = self.product
        
        mu_react = self.reactant.mu
        mu_prod = self.product.mu
        
        reactant_composition = reactant.composition
        product_composition = product.composition
        
        mass_balance = self.compute_mass_balance() 
        # print('reaction:', reactant_composition, '->', product_composition)
        # print('mass_balance', mass_balance)       
        
        fraction = mass_balance['fraction']
        n_H = mass_balance['n_H']
        n_charge = mass_balance['n_charge']
        n_H2O = mass_balance['n_H2O']
        n_L = mass_balance['n_L']
        coeff = self.thermo.kB*self.thermo.T*np.log(10)

        pH_coeff = n_H * coeff
        eU_coeff = -n_charge

        ligand_standard_energy = 0
        
        p_ligand_coeff_dict = {}
        for ligand in n_L:
            n = n_L[ligand]
            delta_mu_ligand = data.mu_ligand[ligand]
            # mu_ligand stores standard-state free energy only.
            # Concentration/activity dependence is handled by the p_ligand term below.
            ligand_standard_energy += n * delta_mu_ligand
            p_ligand_coeff_dict[ligand] = n * coeff

        constants = fraction*mu_react - mu_prod - n_H2O*self.thermo.mu_H2O - ligand_standard_energy
        # constants = fraction*mu_react - mu_prod - n_H2O*self.thermo.mu_H2O
        # constants -= ligand_standard_energy
        return  eU_coeff, pH_coeff, p_ligand_coeff_dict, constants
        

    def calculate_energy_grid_foo(self,grid):
        eU_coeff, pH_coeff, p_ligand_coeff_dict, constants = self.formulate_coefficients()
        
        energy_grid = eU_coeff * grid.V_grid + pH_coeff * grid.pH_grid  - constants
        
        for ligand in p_ligand_coeff_dict:
            coeff = p_ligand_coeff_dict[ligand]
            p_ligand_grid = grid.ligand_grid_dict[ligand]
            energy_grid -= coeff*p_ligand_grid
        return energy_grid
    
    