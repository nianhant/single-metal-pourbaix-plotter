import numpy as np

class Thermodynamics:
    def __init__(self, T = 298.15):
        self.kB = 8.6173e-5  # eV/K
        self.T = T# 348.15 #298.15 # K
        self.mu_H2O = -2.458  # Reference for water
        self.correct_ion=True
    
    def apply_ion_correction(self, species):
        return species.mu + self.kB*self.T*np.log(species.activity)
    
