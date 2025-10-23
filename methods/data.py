class Data:
    def __init__(self, metal,mu_bulk, mu_aqueous_ion, mu_complex, mu_ligand, ligand_concentration, ion_activity, T):
        self.metal = metal
        self.mu_bulk = mu_bulk
        self.mu_aqueous_ion = mu_aqueous_ion
        self.mu_complex = mu_complex
        self.mu_ligand = mu_ligand
        self.ligand_concentration = ligand_concentration
        self.total_solid_species = len(mu_bulk)
        self.total_aq_species = len(mu_aqueous_ion) + len(mu_complex)
        self.ion_activity = ion_activity
        self.T =T

