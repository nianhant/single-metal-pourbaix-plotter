"""Synchronize the verified submission panels back to their project folders."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

from regenerate_submission_2V import ROOT, OUT, SUB

BACKUP = OUT / 'designated_folder_backup_20261011'
SVG = OUT / 'supplementary_svg'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_svg(records):
    import generate_si_single_metal_panels as si
    import generate_sensitivity_triptychs as sweep
    from generate_batch_pourbaix import load_complex_dataframe
    df = load_complex_dataframe()
    for record in records:
        source = Path(record['source'])
        if source.parent.name != 'single' or source.with_suffix('.svg').exists():
            continue
        name = record['name']
        metal = name.split('_')[1 if name.startswith('SI_') else 0]
        print('Generating SVG:', name, flush=True)
        if name.startswith('SI_'):
            cases = [(title, *si.calculate_case(metal, ligands, legacy, df, 1000))
                     for title, ligands, legacy in si.CONDITIONS[metal]]
            si.render(metal, cases, SVG, ['svg'], 600)
        else:
            parameter = name.split('_')[1]
            cases = [(v, *sweep.calculate_case(metal, parameter, v, df, 1000))
                     for v in sweep.LEVELS[parameter]]
            sweep.render_triptych(metal, parameter, cases, SVG, ['svg'], 600)


def destination(record):
    name = record['name']
    single = ROOT / 'examples/figures/figures'
    binary = ROOT.parent / 'Alloy-Pourbaix-Diagram-Plotter/figures/alloy_pourbaix_diagrams'
    if name == 'figure1.pdf':
        return single / 'six_panel_nh3_gly_0p1_no_cn' / 'Au_Ni_Cu_Pd_Pt_Ti_activity_1e-4_NH3_0p02_Gly_0p1_2x3_overwide_side_colors_upper_ticks.pdf'
    if name == 'figure2.pdf':
        return single / 'pd_pt_nh3_gly_0p1_cn_1e-4' / 'Pd_Pt_activity_1e-4_NH3_0p02_Gly_0p1_CN_1e-4_1x2_side_colors.pdf'
    if name == 'figure3.pdf':
        return binary / 'rsc_collage/TiNi_TiCu_PdAu_NH3=0.02M_Gly=0.1M_activity=1e-4M.pdf'
    is_single = Path(record['source']).parent.name == 'single'
    if name.startswith('SI_'):
        return (single / 'si_pourbaix_preview/single_metals' if is_single else binary / 'si_full_legends') / name
    return (single if is_single else binary) / 'sensitivity_triptychs_with_cn_sweep' / name.split('_')[0] / name


def sync(records):
    actions = []
    for record in records:
        source = Path(record['source'])
        assert digest(SUB / record['name']) == record['sha256'], ('Submission changed', record['name'])
        assert digest(source) == record['sha256'], ('Source changed', source)
        target = destination(record)
        assert target.parent.is_dir(), target
        for ext in ('.pdf', '.png', '.svg', '.tif', '.tiff'):
            src = source.with_suffix(ext)
            if ext == '.svg' and not src.exists():
                src = SVG / source.with_suffix(ext).name
            if not src.exists():
                assert not target.with_suffix(ext).exists(), ('Stale export without replacement', target.with_suffix(ext))
                continue
            dst = target.with_suffix(ext)
            actions.append((src, dst))
    assert len({str(dst) for _, dst in actions}) == len(actions)
    assert not BACKUP.exists(), 'Backup already exists; do not overwrite it.'
    BACKUP.mkdir(parents=True)
    manifest = []
    for src, dst in actions:
        old_hash = None
        if dst.exists():
            old_hash = digest(dst)
            backup = BACKUP / dst.relative_to(ROOT.parent)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup)
            assert digest(backup) == old_hash
        shutil.copy2(src, dst)
        assert digest(dst) == digest(src)
        manifest.append(dict(source=str(src), destination=str(dst), previous_sha256=old_hash,
                             sha256=digest(dst)))
    (BACKUP / 'sync_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Updated {len(records)} figures / {len(actions)} exports. Backups: {BACKUP}', flush=True)


if __name__ == '__main__':
    records = json.loads((OUT / 'installation_manifest.json').read_text())
    if '--prepare-svg' in sys.argv:
        prepare_svg(records)
    if '--sync' in sys.argv:
        sync(records)
