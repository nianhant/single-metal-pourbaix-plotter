import numpy as np
import itertools
from reaction import Reaction
class StabilityCalculator:
    def __init__(self, gridplotter, all_species, thermo, data):
        self.grid = gridplotter
        self.all_species = all_species
        self.thermo = thermo
        self.data = data


    def compute_stability(self, dtype=np.float32):
        nx, ny = self.grid.pH_grid.shape

        best_mu = np.full((nx, ny), np.inf, dtype=dtype)
        best_idx = np.full((nx, ny), -1, dtype=np.int32)

        for i, species in enumerate(self.all_species):
            reaction = Reaction(species, species, self.thermo, self.data)
            mu = reaction.calculate_energy_grid(self.grid)
            better = mu < best_mu
            best_mu[better] = mu[better]
            best_idx[better] = i

        return {
            "species_list": self.all_species,
            "stable_index_grid": best_idx,
            "stable_mu_grid": best_mu,
        }

    def compute_stable_regions(self):
        result = self.compute_stability()
        idx = result["stable_index_grid"]
        return {
            species: (idx == i)
            for i, species in enumerate(self.all_species)
        }
    
    # def compute_stable_regions(self):
    #     # stable_regions = {species: np.full((self.grid.grid_size, self.grid.grid_size), True) for species in self.combined_species}
    #     species_list = list(self.combined_species)
    #     n_species = len(species_list)

    #     mu_stack = np.empty((n_species, self.grid.grid_size, self.grid.grid_size))
    #     for i, species in enumerate(species_list):
    #         # print(species.formula)
            
    #         reaction = Reaction(species, species, self.thermo, self.data)
    #         energy_grid = reaction.calculate_energy_grid(self.grid)
    #         # print(energy_grid)
    #         mu_stack[i] = energy_grid

    #     stable_index_grid = np.argmin(mu_stack, axis=0)
    #     stable_regions = {
    #         species: (stable_index_grid == i)
    #         for i, species in enumerate(species_list)
    #     }

    #     return stable_regions
        
        # reactant_product_pairs = [
        #     (reactant, product)
        #     for reactant, product in itertools.combinations(self.combined_species, 2)
        #     if reactant.formula != product.formula or (reactant.formula == product.formula and reactant.phase != product.phase)
        # ]
        # for reactant, product in reactant_product_pairs:
        #     reaction = Reaction(reactant, product, self.thermo, self.data)
        #     energy_grid = reaction.calculate_energy_grid(self.grid)

        #     product_stable_indices = energy_grid < 0  # Product stable when energy < 0
        #     reactant_stable_indices = np.logical_not(product_stable_indices)  # Reactant stable when product is unstable

        #     stable_regions[product] = np.logical_and(stable_regions[product], product_stable_indices)
        #     stable_regions[reactant] = np.logical_and(stable_regions[reactant], reactant_stable_indices)
        
        return stable_regions
        
    # def compute_stable_regions(self):
    #     stable_regions = {species: np.full((self.grid.grid_size, self.grid.grid_size), True) for species in self.combined_species}
    #     reactant_product_pairs = [
    #         (reactant, product)
    #         for reactant, product in itertools.combinations(self.combined_species, 2)
    #         if reactant.formula != product.formula or (reactant.formula == product.formula and reactant.phase != product.phase)
    #     ]
    #     for reactant, product in reactant_product_pairs:
    #         reaction = Reaction(reactant, product, self.thermo, self.data)
    #         energy_grid = reaction.calculate_energy_grid(self.grid)

    #         product_stable_indices = energy_grid < 0  # Product stable when energy < 0
    #         reactant_stable_indices = np.logical_not(product_stable_indices)  # Reactant stable when product is unstable

    #         stable_regions[product] = np.logical_and(stable_regions[product], product_stable_indices)
    #         stable_regions[reactant] = np.logical_and(stable_regions[reactant], reactant_stable_indices)
        
    #     return stable_regions