"""Verify CN sweep changes using phase identities, not image pixels."""
import sys
import json
from pathlib import Path
import numpy as np

root = Path('/global/homes/n/nianhant/data/stability_paper/single-metal-pourbaix-plotter')
sys.path.insert(0, str(root / 'examples'))
import generate_sensitivity_triptychs as sweep

df = sweep.load_complex_dataframe()
records = []
for metal in sweep.METALS:
    maps = []
    for level in sweep.LEVELS['CN']:
        _, regions, _ = sweep.calculate_case(metal, 'CN', level, df, sweep.GRID_SIZE)
        phase_map = np.full(next(iter(regions.values())).shape, '', dtype=object)
        for species, mask in regions.items():
            phase_map[mask] = species.phase + ':' + species.formula
        maps.append(phase_map)
    changed = [float(np.mean(grid != maps[0])) for grid in maps]
    record = {'metal': metal, 'grid_size': sweep.GRID_SIZE,
              'CN_M': list(sweep.LEVELS['CN']), 'changed_fraction': changed,
              'include': any(value > 0 for value in changed)}
    records.append(record)
    print(json.dumps(record), flush=True)
Path('/tmp/cn_sensitivity_audit.json').write_text(json.dumps(records, indent=2))
