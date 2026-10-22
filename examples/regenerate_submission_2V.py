"""Restore the submission reaction-window overlays, preserving panel settings.

Run with base-agent Python; pass single or binary as the first argument.
Outputs stay outside the manuscript repository until separately reviewed.
"""
from pathlib import Path
import sys
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'examples/figures/figures/potential_window_2V_grid1000_20261006'
SUB = ROOT.parent / 'updated_stability_manuscript/submission'


def single():
    import generate_six_panel_pourbaix as six
    import generate_pd_pt_cn_panel as pair
    import generate_si_single_metal_panels as si
    import generate_sensitivity_triptychs as sweep
    from matplotlib.figure import Figure
    original = Figure.savefig
    def save(fig, fname, *args, **kwargs):
        if Path(fname).suffix in ('.tif', '.tiff'):
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        return original(fig, fname, *args, **kwargs)
    Figure.savefig = save
    output = OUT / 'single'
    output.mkdir(parents=True, exist_ok=True)
    df = six.load_complex_dataframe()
    cases = []
    for metal in six.METALS:
        data, thermo, species, labels = six.prepare_case(
            metal, six.ACTIVITY, six.TEMPERATURE, six.LIGAND_CONCENTRATIONS,
            df, ligand_correction=False)
        plotter = six.GridPlotter(pH_range=six.PH_RANGE, V_range=six.POTENTIAL_RANGE,
            data=data, grid_size=six.GRID_SIZE, save_fig=False, dir=str(output), filename='figure1', dpi=six.DPI)
        regions = six.StabilityCalculator(plotter, species, thermo, data).compute_stable_regions()
        cases.append((metal, plotter, regions, labels))
    six.render_side_legends(cases, columns=3, overwide=True, axes_mode='upper_ticks',
        colored_only=True, output_dir=output, output_stem='figure1')
    pair.OUTPUT_DIR = output
    pair.OUTPUT_STEM = 'figure2'
    pair.main()
    for metal in si.METALS:
        print('SI', metal, flush=True)
        cases = [(title, *si.calculate_case(metal, ligands, legacy, df, 1000))
                 for title, ligands, legacy in si.CONDITIONS[metal]]
        si.render(metal, cases, output, ['pdf', 'png', 'tif'], 600)
    text = (SUB / 'pourbaix_sensitivity_SI.tex').read_text()
    for metal, parameter in re.findall(r'\{([A-Za-z]+)_(activity|NH3|Gly|CN)_three_levels', text):
        print('Sensitivity', metal, parameter, flush=True)
        cases = [(value, *sweep.calculate_case(metal, parameter, value, df, 1000))
                 for value in sweep.LEVELS[parameter]]
        sweep.render_triptych(metal, parameter, cases, output, ['pdf', 'png', 'tif'], 600)


def binary():
    import hashlib
    import numpy as np
    project = ROOT.parent / 'Alloy-Pourbaix-Diagram-Plotter'
    sys.path.insert(0, str(project))
    from examples import make_rsc_pourbaix_collage as collage
    from examples import render_si_alloy_full_legends as full
    from examples import generate_alloy_sensitivity_triptychs as sweep
    from methods.data_loader import SpeciesDataLoader
    from methods.grid import GridMaker
    from methods.thermodynamics import PourbaixData, SpeciesGridResult
    from methods.visualization import PlotAccessories
    output = OUT / 'binary'
    output.mkdir(parents=True, exist_ok=True)
    collage.OUTPUT_DIR = output
    collage.OUTPUT_STEM = 'figure3'
    collage.main()
    si_manifest = json.loads((full.si.RESULTS / 'manifest.json').read_text())
    groups = {g['alloy']: g for g in json.loads((full.si.RESULTS / 'source_inventory.json').read_text())}
    for item in si_manifest['figures']:
        alloy = item['alloy']
        data = SpeciesDataLoader(*groups[alloy]['metals'], collage.MU_LIGAND,
            collage.TEMPERATURE_K, data_dir=collage.DATA_DIR)
        accessories = PlotAccessories(data)
        print('Recalculating SI', alloy, '1000 x 1000', flush=True)
        cases = [sweep.calculate_case(data, condition, 1000) for condition in item['conditions']]
        colors = sweep.consistent_colors(cases, accessories)
        sweep.render(alloy, 'CN', cases, accessories, colors, output,
            titles=[''] * len(cases), output_stem=item['stem'] + '_all_species', include_aqueous=True)
    source = project / 'research/2026-10-02_alloy_concentration_sensitivity'
    manifest = json.loads((source / 'manifest.json').read_text())
    for path, digest in manifest['data_sha256'].items():
        assert hashlib.sha256((project / path).read_bytes()).hexdigest() == digest
    text = (SUB / 'alloy_pourbaix_sensitivity_SI.tex').read_text()
    included = re.findall(r'\{([A-Za-z]+)_(activity|NH3|Gly|CN)_three_levels', text)
    records = []
    for m1, m2, alloy in collage.SYSTEMS:
        data = SpeciesDataLoader(m1, m2, collage.MU_LIGAND, collage.TEMPERATURE_K, data_dir=collage.DATA_DIR)
        accessories = PlotAccessories(data)
        cache = {}
        for path in sorted((source / alloy).glob('case_*.json')):
            saved = json.loads(path.read_text())
            condition = saved['condition']
            print('Recalculating sensitivity', alloy, condition, flush=True)
            case = sweep.calculate_case(data, condition, 1000)
            cache[sweep.condition_key(condition)] = case
            audit = output / 'grids' / alloy
            audit.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(audit / path.with_suffix('.npz').name,
                stable_indices=case['regions'].min_indices,
                pH=case['grid'].pH_values, E_SHE=case['grid'].V_values)
        colors = sweep.consistent_colors(list(cache.values()), accessories)
        baseline = cache[sweep.condition_key(sweep.BASELINE)]
        for name, parameter in included:
            if name != alloy:
                continue
            cases = [cache[sweep.condition_key({**sweep.BASELINE, parameter: v})] for v in sweep.LEVELS[parameter]]
            palette = sweep.consistent_colors(cases, accessories) if parameter == 'CN' else colors
            sweep.render(alloy, parameter, cases, accessories, palette, output, include_aqueous=parameter == 'CN')
            for case in cases:
                records.append(dict(alloy=alloy, parameter=parameter, **case['condition'],
                                    **sweep.summarize(case, baseline, accessories)))
        sweep.write_csv(output / 'sensitivity_summary_2V.csv', records)
    manifest['experimental_E_RHE'] = list(collage.V_EXP_RANGE)
    manifest['grid_size'] = 1000
    manifest['note'] = 'Recalculated at 1000 x 1000 with restored +2 V reaction-window upper bound.'
    (output / 'window_provenance.json').write_text(json.dumps(manifest, indent=2))
    path = output / 'si_grid_manifest.json'
    si_manifest['grid_size'] = 1000
    si_manifest['experimental_E_RHE'] = list(collage.V_EXP_RANGE)
    path.write_text(json.dumps(si_manifest, indent=2))


if __name__ == '__main__':
    {'single': single, 'binary': binary}[sys.argv[1]]()
