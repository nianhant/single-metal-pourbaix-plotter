"""Replace existing -2 to +2 V RHE boxes in saved SVGs with +2 +/-0.1 V.

Review artwork only: phase maps and publication sources are not modified.
Uses the original box geometry, including its existing pH interval and SHE
conversion. Requires exactly four clipped black line segments per panel.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/figures/figures/potential_window_2V_grid1000_20261006'
OUT = ROOT / 'research/reaction_conditions_review'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


def preview(source, destination):
    tree = ET.parse(source)
    root = tree.getroot()
    count = 0
    for axes in root.findall(f'.//{{{NS}}}g'):
        if not axes.get('id', '').startswith('axes_'):
            continue
        segments = []
        for group in axes.findall(f'{{{NS}}}g'):
            if not group.get('id', '').startswith('line2d_'):
                continue
            for path in group.findall(f'{{{NS}}}path'):
                if path.get('clip-path') and 'stroke: #000000;' in path.get('style', ''):
                    values = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', path.get('d', ''))]
                    if len(values) == 4:
                        segments.append((group, path, values))
        if not segments:
            continue
        assert len(segments) == 4, (source, axes.get('id'), len(segments))
        sloped = [s for s in segments if s[2][0] != s[2][2]]
        assert len(sloped) == 2
        top, bottom = sorted(sloped, key=lambda s: s[2][1])
        x1, y1, x2, y2 = top[2]
        pixels_per_volt = (bottom[2][1] - y1) / 4.0
        # Confirm the original interval is pH 10.5--13.5, using its slope.
        assert abs((y2 - y1) / pixels_per_volt - 0.0591 * 3) < 1e-5
        clip = top[1].get('clip-path')
        for group, path, values in segments:
            axes.remove(group)
        band = ET.SubElement(axes, f'{{{NS}}}g', {'id': f'reaction_condition_{count}'})
        d = 0.1 * pixels_per_volt
        points = f'{x1},{y1-d} {x2},{y2-d} {x2},{y2+d} {x1},{y1+d}'
        ET.SubElement(band, f'{{{NS}}}polygon', {
            'points': points, 'fill': '#009e73', 'fill-opacity': '0.30', 'clip-path': clip})
        ET.SubElement(band, f'{{{NS}}}path', {
            'd': f'M {x1} {y1} L {x2} {y2}', 'clip-path': clip,
            'style': 'fill: none; stroke: #000000; stroke-width: 1.1'})
        count += 1
    assert count > 0
    view = [float(v) for v in root.get('viewBox').split()]
    root.set('viewBox', f'0 0 {view[2]} {view[3]+28}')
    root.set('height', f'{view[3]+28}pt')
    for offset, line in enumerate([
        'PREVIEW: black = 2.0 V RHE; green = +/-0.1 V.',
        'pH 10.5-13.5 retained as conservative envelope; not measured trajectory.',
    ]):
        text = ET.SubElement(root, f'{{{NS}}}text', {
            'x': '12', 'y': str(view[3] + 11 + offset * 10),
            'font-size': '7', 'font-family': 'sans-serif', 'fill': '#000000'})
        text.text = line
    tree.write(destination, encoding='utf-8', xml_declaration=True)
    print(f'{destination}: {count} panels')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in [
        ('single_metals', BASE / 'single/figure1_2x3_overwide_side_colors_upper_ticks.svg'),
        ('pd_pt_cyanide', BASE / 'single/figure2_1x2_side_colors.svg'),
        ('binary_metals', BASE / 'binary/figure3.svg'),
    ]:
        preview(source, OUT / f'{name}_reaction_conditions.svg')
