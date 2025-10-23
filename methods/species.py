import numpy as np
import copy

import re
from ase.symbols import string2symbols
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

class Species:
    def __init__(self, formula, metal,thermo, data, phase, activity):
        self.formula = formula
        self.metal = metal
        self.thermo = thermo
        self.data = data
        self.phase = phase #'bulk', 'aqueous_ion','ligand','complex'
        self.activity = activity
        self.ligand = None
        self.mu = self.get_mu()
        if self.phase == 'complex':
            self.composition = self.parse_complex_composition()
            
        else:
            self.composition = self.parse_formula_composition()
        if self.phase =='ligand':
            self.activity = data.ligand_concentration[formula]
        
    
    def parse_formula_composition(self):
        formula = self.formula.split('(aq)')[0]
        composition = {'charge': 0,
                       'H': 0,
                       'O': 0,
                       self.metal:0}
        ############## count n of electrons and protons ##############
        match = re.search(r'\[(\d*)([+-])\]', formula)
        if match:
            charge = match.group(1)
            sign = match.group(2)
            composition['charge'] = int(sign + charge)

        ############## count n of O and H ##############
        element_only_form = formula.split('[')[0]
        symbol_list = string2symbols(element_only_form)
        for symbol in symbol_list:
            if symbol in composition:
                composition[symbol] += 1
        return composition
    
    def parse_complex_composition(self):
        composition = {'H': 0,
                       'O': 0,
                       'charge': 0,
                       self.metal:0}
        ############## count n of electrons and protons ##############
        match = re.search(r'\[(\d*)([+-])\]', self.formula)
        if match:
            charge = match.group(1)
            sign = match.group(2)
            composition['charge'] = int(sign + charge)
        ############## count n of ligands ##############
        pattern = r'([A-Za-z]+)?\(?([A-Za-z0-9]+)\)?(\d*)'
        match = re.search(pattern, self.formula)
        if match:
            metal = match.group(1)      # Metal symbol (e.g., Ni)
            ligand = match.group(2)     # Ligand (e.g., NH3)
            ligand_count = match.group(3)  # Ligand count (e.g., 4, could be empty)

            # If the ligand count is empty, it means it's 1
            ligand_count = int(ligand_count) if ligand_count else 1
        composition[ligand] = ligand_count
        self.ligand = ligand
        ############## count n of metal ##############
        element_only_form = self.formula.split('(')[0]
        symbol_list = string2symbols(element_only_form)
        for symbol in symbol_list:
            if symbol in composition:
                composition[symbol] += 1
        return composition
        
    def get_mu(self):
        if self.phase == 'bulk':
            mu = self.data.mu_bulk[self.formula]
        elif self.phase == 'aqueous_ion':
            mu = self.data.mu_aqueous_ion[self.formula]
            self.mu = mu
            if self.thermo.correct_ion:
                mu = self.thermo.apply_ion_correction(self)
        elif self.phase == 'complex':
            mu = self.data.mu_complex[self.formula]
            self.mu = mu
            if self.thermo.correct_ion:
                mu = self.thermo.apply_ion_correction(self)
        return mu
    
