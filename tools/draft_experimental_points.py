"""Draft experimental endpoint overlays on existing Fig. 1 and Fig. 3 SVGs.

Requires cairosvg for rendering. pH values are visual estimates from si_pH.png,
not original measurements; all potentials are nominal 2.0 V RHE for layout.
"""
from pathlib import Path
import csv
import os
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/figures/figures/potential_window_2V_grid1000_20261006'
OUT = ROOT / 'research/reaction_conditions_review/endpoint_drafts'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
PH = {'Au': (11.15, 11.03), 'Ni': (11.30, 10.84),
      'Cu': (11.37, 10.75), 'Pd': (11.49, 10.86),
      'TiNi': (12.03, 11.49), 'TiCu': (11.59, 11.14),
      'PdAu': (11.71, 10.71)}


def element(parent, tag, **attrs):
    return ET.SubElement(parent, f'{{{NS}}}{tag}',
                         {k.replace('_', '-'): str(v) for k, v in attrs.items()})


def marker(parent, x, y, final=False):
    # Use equal size; fill alone distinguishes the two sampling times.
    radius = 2.6
    element(parent, 'circle', cx=x, cy=y, r=radius + .65,
            fill='white', stroke='none')
    element(parent, 'circle', cx=x, cy=y, r=radius,
            fill='black' if final else 'white', stroke='black', stroke_width=.8)


def draft(source, name, metals):
    import cairosvg
    tree = ET.parse(source)
    root = tree.getroot()
    panel = 0
    for axes in root.findall(f'.//{{{NS}}}g'):
        if not axes.get('id', '').startswith('axes_'):
            continue
        segments = []
        for group in axes.findall(f'{{{NS}}}g'):
            if not group.get('id', '').startswith('line2d_'):
                continue
            for path in group.findall(f'{{{NS}}}path'):
                if path.get('clip-path') and 'stroke: #000000;' in path.get('style', ''):
                    values = list(map(float, re.findall(r'-?\d+(?:\.\d+)?', path.get('d', ''))))
                    if len(values) == 4:
                        segments.append((group, path, values))
        if not segments:
            continue
        assert len(segments) == 4
        top, bottom = sorted([s for s in segments if s[2][0] != s[2][2]], key=lambda s: s[2][1])
        x1, y1, x2, y2 = top[2]
        units_per_volt = (bottom[2][1] - y1) / 4
        assert abs((y2-y1)/units_per_volt - .0591*3) < 1e-5
        # Old upper edge is 2.0 V RHE and spans pH 10.5--13.5.
        def position(ph):
            fraction = (ph-10.5)/3
            return x1 + fraction*(x2-x1), y1 + fraction*(y2-y1)
        for _, path, _ in segments:
            path.set('style', 'fill: none; stroke: #777777; stroke-width: 0.6; '
                     'stroke-dasharray: 3,2; stroke-linecap: butt')
        metal = metals[panel]
        panel += 1
        if metal not in PH:
            continue
        overlay = element(axes, 'g', id=f'experimental-endpoints-{metal}',
                          clip_path=top[1].get('clip-path'))
        start, finish = map(position, PH[metal])
        marker(overlay, *finish, final=True)
        # Preserve the open outline when the endpoint coordinates overlap.
        marker(overlay, *start)
    assert panel == len(metals)
    width, height = map(float, root.get('viewBox').split()[2:])
    # Position relative to the actual x-axis labels, not the source canvas margin.
    label_baselines = [float(t.get('y')) for t in root.iter(f'{{{NS}}}text')
                       if ''.join(t.itertext()) == 'pH' and t.get('y')]
    if not label_baselines:
        # Fig. 3 stores its labels as glyph paths with a translated baseline.
        label_baselines = [float(y) for y in re.findall(
            r'<!-- pH -->\s*<g transform="translate\([\d.]+ ([\d.]+)\)',
            source.read_text())]
    assert label_baselines, 'Cannot locate pH labels for legend placement'
    legend_y = max(label_baselines) + 9
    new_height = legend_y + 7
    root.set('viewBox', f'0 0 {width} {new_height}')
    root.set('height', f'{new_height}pt')
    if new_height > height:
        element(root, 'rect', x=0, y=height, width=width,
                height=new_height-height, fill='white')
    key = element(root, 'g', id='experimental-key')
    x, y = width/2-51, legend_y
    marker(key, x, y)
    element(key, 'text', x=x+8, y=y+2.5, font_family='DejaVu Sans', font_size=7).text='Initial'
    marker(key, x+60, y, final=True)
    element(key, 'text', x=x+68, y=y+2.5, font_family='DejaVu Sans', font_size=7).text='120 min'
    element(root, 'desc').text=(
        'Draft coordinates: pH visually estimated from si_pH.png; nominal '
        '2.0 V RHE at both endpoints for every electrode, including Cu. '
        'Dashed boxes show the reference envelope, pH 10.5-13.5 and '
        '-2 to +2 V RHE. Endpoints are shown without connecting lines.')
    target = OUT / name
    tree.write(target.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)
    cairosvg.svg2pdf(url=str(target.with_suffix('.svg')), write_to=str(target.with_suffix('.pdf')))
    cairosvg.svg2png(url=str(target.with_suffix('.svg')), write_to=str(target.with_suffix('.png')), scale=3, background_color='white')
    print(target)


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/pourbaix-matplotlib-cache')
    import matplotlib
    OUT.mkdir(parents=True, exist_ok=True)
    # Supply the same DejaVu fonts Matplotlib used for the source artwork.
    config = ET.Element('fontconfig')
    ET.SubElement(config, 'include').text = '/etc/fonts/fonts.conf'
    ET.SubElement(config, 'dir').text = str(Path(matplotlib.get_data_path())/'fonts/ttf')
    ET.SubElement(config, 'cachedir').text = '/tmp/pourbaix-font-cache'
    font_config = OUT/'fonts.conf'
    ET.ElementTree(config).write(font_config, encoding='utf-8')
    os.environ['FONTCONFIG_FILE'] = str(font_config)
    with (OUT/'approximate_endpoints.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['electrode','endpoint','pH_estimated_from_SI_image','E_RHE_nominal_V','E_SHE_for_draft_V'])
        for metal, endpoints in PH.items():
            for label, ph in zip(['Blank','120 min'], endpoints):
                writer.writerow([metal,label,ph,2.0,round(2-.0591*ph,6)])
    draft(BASE/'single/figure1_2x3_overwide_side_colors_upper_ticks.svg',
          'Fig1_experimental_endpoints_DRAFT', ['Au','Ni','Cu','Pd','Pt','Ti'])
    draft(BASE/'binary/figure3.svg', 'Fig3_experimental_endpoints_DRAFT',
          ['TiNi','TiCu','PdAu'])
