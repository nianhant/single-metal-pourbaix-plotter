import json
import os
from typing import Optional

import re


def get_mpr(api_key: Optional[str] = None):
    try:
        from mp_api.client import MPRester
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("mp_api is required to fetch uncached Materials Project data.") from exc

    key = api_key or os.getenv("MP_API_KEY") or os.getenv("MAPI_KEY")
    if not key:
        raise ValueError(
            "No Materials Project API key provided. "
            "Pass `api_key` or set MP_API_KEY / MAPI_KEY in your environment."
        )
    return MPRester(key)

# Ensure 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

def save_as_json(data, filename):
    with open(f'data/{filename}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

        
def get_ion_formation_energy(metal, mpr, save_json=True):
    ion_formation_eng = {}
    
    pourbaix_entries = mpr.get_pourbaix_entries([metal])
    for entry in pourbaix_entries:
        formula = entry.name
        if entry.phase_type == 'Ion':
            trimmed_formula = formula
            trimmed_formula = re.sub(r'\[([+-])(\d+)\]', r'[\2\1]', trimmed_formula)
            if trimmed_formula not in ion_formation_eng:
                ion_formation_eng[trimmed_formula] = entry.uncorrected_energy
            elif ion_formation_eng[trimmed_formula] > entry.uncorrected_energy:
                ion_formation_eng[trimmed_formula] = entry.uncorrected_energy
    if save_json:
        save_as_json(ion_formation_eng, f'{metal}_ion_formation_energy')
    
    return ion_formation_eng



def get_solid_formation_energy(metal, mpr, stable_only=False, save_json=True):
    try:
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.entries.computed_entries import ComputedEntry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pymatgen is required to fetch uncached solid formation energies.") from exc

    # returns per metal formation energy, not total
    solid_formation_eng = {}    
    pourbaix_entries = mpr.get_pourbaix_entries([metal])

    solid_entries = [entry for entry in pourbaix_entries if entry.phase_type == "Solid"]
    entries_HO = [ComputedEntry("H", 0), ComputedEntry("O", 2.46)]
    solid_pd = PhaseDiagram(solid_entries + entries_HO)
    if stable_only:
        solid_entries = list(set(solid_pd.stable_entries) - set(entries_HO))
    else:
        solid_entries = list(set(solid_pd.entries) - set(entries_HO))
    
    for entry in solid_entries:
        formula = entry.composition.formula
        trimmed_formula = formula.replace(" ", "")
        trimmed_formula = re.sub(r'([A-Za-z])1(?=[A-Za-z]|$)', r'\1',trimmed_formula)
        if trimmed_formula not in solid_formation_eng:
            solid_formation_eng[trimmed_formula] = entry.uncorrected_energy
        elif solid_formation_eng[trimmed_formula] > entry.uncorrected_energy:
            solid_formation_eng[trimmed_formula] = entry.uncorrected_energy
    if save_json:
        save_as_json(solid_formation_eng, f'{metal}_solid_formation_energy')
    return solid_formation_eng
