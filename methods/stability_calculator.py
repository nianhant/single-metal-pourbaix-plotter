import numpy as np
import itertools
from reaction import Reaction
class StabilityCalculator:
    def __init__(self, gridplotter, combined_species, thermo, data):
        self.grid = gridplotter
        self.combined_species = combined_species
        self.thermo = thermo
        self.data = data

    def compute_stable_regions(self):
        stable_regions = {species: np.full((self.grid.grid_size, self.grid.grid_size), True) for species in self.combined_species}
        reactant_product_pairs = [
            (reactant, product)
            for reactant, product in itertools.combinations(self.combined_species, 2)
            if reactant.formula != product.formula or (reactant.formula == product.formula and reactant.phase != product.phase)
        ]
        for reactant, product in reactant_product_pairs:
            reaction = Reaction(reactant, product, self.thermo, self.data)
            energy_grid = reaction.calculate_energy_grid(self.grid)

            product_stable_indices = energy_grid < 0  # Product stable when energy < 0
            reactant_stable_indices = np.logical_not(product_stable_indices)  # Reactant stable when product is unstable

            stable_regions[product] = np.logical_and(stable_regions[product], product_stable_indices)
            stable_regions[reactant] = np.logical_and(stable_regions[reactant], reactant_stable_indices)
        
        return stable_regions