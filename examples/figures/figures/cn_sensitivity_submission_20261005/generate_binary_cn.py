"""Regenerate CN-only binary sweeps with existing publication renderer."""
import sys
import json
from pathlib import Path

root = Path('/global/homes/n/nianhant/data/stability_paper/Alloy-Pourbaix-Diagram-Plotter')
sys.path.insert(0, str(root / 'examples'))
import generate_alloy_sensitivity_triptychs as sweep

records = []
for metal1, metal2, alloy in sweep.collage.SYSTEMS:
    data = sweep.SpeciesDataLoader(metal1, metal2, sweep.collage.MU_LIGAND,
                                  sweep.collage.TEMPERATURE_K, data_dir=sweep.collage.DATA_DIR)
    accessories = sweep.PlotAccessories(data)
    cases = [sweep.calculate_case(data, {**sweep.BASELINE, 'CN': value}, 1600)
             for value in sweep.LEVELS['CN']]
    stats = [sweep.summarize(case, cases[0], accessories) for case in cases]
    changed = [row['full_changed_fraction'] for row in stats]
    record = {'alloy': alloy, 'CN_M': list(sweep.LEVELS['CN']),
              'grid_size': 1600, 'changed_fraction': changed,
              'include': any(value > 0 for value in changed)}
    records.append(record)
    print(json.dumps(record), flush=True)
    if record['include']:
        colors = sweep.consistent_colors(cases, accessories)
        sweep.render(alloy, 'CN', cases, accessories, colors,
                     Path('/tmp/cn-si-edit/binary_figures') / alloy, include_aqueous=True)
Path('/tmp/cn-si-edit/binary_audit.json').write_text(json.dumps(records, indent=2))
