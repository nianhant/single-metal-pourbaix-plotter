from dataclasses import dataclass

import numpy as np

from reaction import Reaction


@dataclass(frozen=True)
class StabilityResult:
    species_list: list
    stable_index_grid: np.ndarray
    stable_mu_grid: np.ndarray


class StabilityCalculator:
    def __init__(self, gridplotter, all_species, thermo, data):
        self.grid = gridplotter
        self.all_species = list(all_species)
        self.thermo = thermo
        self.data = data
        self._self_reactions = [
            Reaction(species, species, self.thermo, self.data)
            for species in self.all_species
        ]

    def _energy_grid_shape(self):
        return self.grid.pH_grid.shape

    def _compute_species_energy_stack(self, dtype):
        shape = (len(self._self_reactions),) + self._energy_grid_shape()
        energy_stack = np.empty(shape, dtype=dtype)

        for index, reaction in enumerate(self._self_reactions):
            energy_stack[index] = reaction.calculate_energy_grid(self.grid)

        return energy_stack

    def compute_stability(self, dtype=np.float32):
        if not self.all_species:
            empty_shape = self._energy_grid_shape()
            return StabilityResult(
                species_list=[],
                stable_index_grid=np.full(empty_shape, -1, dtype=np.int32),
                stable_mu_grid=np.full(empty_shape, np.inf, dtype=dtype),
            )

        energy_stack = self._compute_species_energy_stack(dtype=dtype)
        stable_index_grid = np.argmin(energy_stack, axis=0).astype(np.int32, copy=False)
        stable_mu_grid = np.min(energy_stack, axis=0)

        return StabilityResult(
            species_list=self.all_species,
            stable_index_grid=stable_index_grid,
            stable_mu_grid=stable_mu_grid,
        )

    def compute_stable_regions(self):
        result = self.compute_stability()
        idx = result.stable_index_grid
        return {
            species: (idx == i)
            for i, species in enumerate(self.all_species)
        }
