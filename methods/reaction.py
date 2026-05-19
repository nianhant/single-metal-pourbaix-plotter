from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReactionCoefficients:
    eU_coeff: float
    pH_coeff: float
    p_ligand_coeff_dict: dict
    constant: float
    metal_normalization: float

class Reaction:
    def __init__(self, reactant, product, thermo, data):
        self.reactant = reactant
        self.product = product
        self.thermo = thermo
        self.data = data
        self._self_energy_coefficients = None

    @property
    def coeff(self):
        return self.thermo.kB * self.thermo.T * np.log(10)

    def _scaled_composition(self, species, scale=1.0):
        return {
            key: value * scale
            for key, value in species.composition.items()
        }

    def _ligand_term(self, species, grid):
        if not species.ligand:
            return 0.0, 0.0

        ligand = species.ligand
        n_ligand = species.composition[ligand]
        ligand_standard_energy = n_ligand * self.data.mu_ligand[ligand]
        p_ligand_grid = grid.ligand_grid_dict[ligand]
        return ligand_standard_energy, n_ligand * self.coeff * p_ligand_grid

    def self_energy_coefficients(self):
        if self._self_energy_coefficients is not None:
            return self._self_energy_coefficients

        composition = self.reactant.composition
        n_h2o = composition["O"]
        n_h = composition["H"] - 2 * n_h2o
        n_eu = composition["charge"] - n_h
        n_metal = composition[self.reactant.metal]

        ligand_standard_energy = 0.0
        p_ligand_coeff_dict = {}
        if self.reactant.ligand:
            ligand = self.reactant.ligand
            n_ligand = composition[ligand]
            ligand_standard_energy = n_ligand * self.data.mu_ligand[ligand]
            p_ligand_coeff_dict[ligand] = n_ligand * self.coeff

        self._self_energy_coefficients = ReactionCoefficients(
            eU_coeff=-n_eu,
            pH_coeff=n_h * self.coeff,
            p_ligand_coeff_dict=p_ligand_coeff_dict,
            constant=self.reactant.mu - n_h2o * self.thermo.mu_H2O - ligand_standard_energy,
            metal_normalization=n_metal,
        )
        return self._self_energy_coefficients

    def calculate_energy_grid(self, grid):
        if self.reactant is self.product:
            coefficients = self.self_energy_coefficients()
            energy_grid = (
                coefficients.eU_coeff * grid.V_grid
                + coefficients.pH_coeff * grid.pH_grid
                + coefficients.constant
            )
            for ligand, coeff in coefficients.p_ligand_coeff_dict.items():
                energy_grid += coeff * grid.ligand_grid_dict[ligand]
            return energy_grid / coefficients.metal_normalization

        ligand_standard_energy, ligand_grid_term = self._ligand_term(self.reactant, grid)
        composition = self.reactant.composition
        n_h2o = composition["O"]
        n_h = composition["H"] - 2 * n_h2o
        n_eu = composition["charge"] - n_h
        n_metal = composition[self.reactant.metal]

        energy_grid = (
            self.reactant.mu
            + n_h * self.coeff * grid.pH_grid
            - n_eu * grid.V_grid
            - n_h2o * self.thermo.mu_H2O
            - ligand_standard_energy
            + ligand_grid_term
        )
        return energy_grid / n_metal

        
    def compute_mass_balance(self):
        reactant = self.reactant 
        product = self.product
        # return a dictionary of mass balances
        rea_composition = dict(self.reactant.composition)
        prod_composition = dict(self.product.composition)

        ############## Balance metal elements ##############
        fraction = prod_composition[product.metal] / rea_composition[reactant.metal]
        rea_composition = self._scaled_composition(self.reactant, scale=fraction)

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
        return {
            'fraction': fraction,
            'n_H2O': n_H2O,
            'n_charge': n_charge,
            'n_H': n_H,
            'n_L': n_L,
        }
    
    def formulate_coefficients(self):
        """
        Calculate the chemical potential difference for a single reaction.
        """
        data = self.data
        reactant = self.reactant
        product = self.product
        
        mu_react = self.reactant.mu
        mu_prod = self.product.mu
        
        mass_balance = self.compute_mass_balance() 
        
        fraction = mass_balance['fraction']
        n_H = mass_balance['n_H']
        n_charge = mass_balance['n_charge']
        n_H2O = mass_balance['n_H2O']
        n_L = mass_balance['n_L']
        coeff = self.coeff

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

        constant = fraction * mu_react - mu_prod - n_H2O * self.thermo.mu_H2O - ligand_standard_energy
        return ReactionCoefficients(
            eU_coeff=eU_coeff,
            pH_coeff=pH_coeff,
            p_ligand_coeff_dict=p_ligand_coeff_dict,
            constant=constant,
            metal_normalization=1.0,
        )
        

    def calculate_energy_grid_foo(self,grid):
        coefficients = self.formulate_coefficients()
        
        energy_grid = (
            coefficients.eU_coeff * grid.V_grid
            + coefficients.pH_coeff * grid.pH_grid
            - coefficients.constant
        )
        
        for ligand, coeff in coefficients.p_ligand_coeff_dict.items():
            energy_grid -= coeff * grid.ligand_grid_dict[ligand]
        return energy_grid
    
    
