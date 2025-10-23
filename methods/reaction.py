import copy
import numpy as np 
import re

class Reaction:
    def __init__(self, reactant, product, thermo, data):
        self.reactant = reactant
        self.product = product
        self.thermo = thermo
        self.data = data
    
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

        ############## Balance ligand ##############
        if not reactant.ligand and not product.ligand:
            n_L = {}
        elif not reactant.ligand: # product has ligand
            n_L = {product.ligand: - prod_composition[product.ligand]}
        elif not product.ligand: # rea has ligand
            n_L = {reactant.ligand: rea_composition[reactant.ligand]}
        if reactant.ligand != None and product.ligand != None :
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
        
        fraction = mass_balance['fraction']
        n_H = mass_balance['n_H']
        n_charge = mass_balance['n_charge']
        n_H2O = mass_balance['n_H2O']
        n_L = mass_balance['n_L']
        coeff = self.thermo.kB*self.thermo.T*np.log(10)

        pH_coeff = n_H * coeff
        eU_coeff = -n_charge

        total_ligand_mu = 0
        
        p_ligand_coeff_dict = {}
        for ligand in n_L:
            n = n_L[ligand]
            delta_mu_ligand = data.mu_ligand[ligand]
            total_ligand_mu += n * delta_mu_ligand # takes care of activity
            p_ligand_coeff_dict[ligand] = n * coeff

        constants = fraction*mu_react - mu_prod - n_H2O*self.thermo.mu_H2O  - total_ligand_mu
        return  eU_coeff, pH_coeff, p_ligand_coeff_dict, constants
        

    def calculate_energy_grid(self,grid):
        eU_coeff, pH_coeff, p_ligand_coeff_dict, constants = self.formulate_coefficients()
        
        energy_grid = eU_coeff * grid.V_grid + pH_coeff * grid.pH_grid  - constants
        
        for ligand in p_ligand_coeff_dict:
            coeff = p_ligand_coeff_dict[ligand]
            p_ligand_grid = grid.ligand_grid_dict[ligand]
            energy_grid -= coeff*p_ligand_grid
        return energy_grid
            