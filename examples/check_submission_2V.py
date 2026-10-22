"""Validate, preview, and optionally install the restored-window PDFs."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import fitz
from PIL import Image, ImageDraw
from regenerate_submission_2V import OUT, SUB

names = ['figure1.pdf', 'figure2.pdf', 'figure3.pdf']
for tex in ('pourbaix_figures_SI.tex', 'pourbaix_sensitivity_SI.tex', 'alloy_pourbaix_sensitivity_SI.tex'):
    names.extend(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+\.pdf)\}', (SUB / tex).read_text()))
assert len(names) == len(set(names)) == 49
records = []
thumbs = []
for name in names:
    if name in ('figure1.pdf', 'figure2.pdf'):
        found = list((OUT / 'single').glob(Path(name).stem + '*.pdf'))
    else:
        found = list(OUT.glob('*/' + name))
    assert len(found) == 1, (name, found)
    source = found[0]
    old = fitz.open(SUB / name)
    new = fitz.open(source)
    assert len(new) == len(old) == 1
    dims_old, dims_new = list(old[0].rect), list(new[0].rect)
    assert max(abs(a-b) for a,b in zip(dims_old, dims_new)) < 0.1, (name, dims_old, dims_new)
    pix = new[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    im.thumbnail((1000, 470))
    thumb = Image.new('RGB', (1020, 510), 'white')
    thumb.paste(im, ((1020-im.width)//2, 30))
    ImageDraw.Draw(thumb).text((10, 8), name, fill='black')
    thumbs.append(thumb)
    tiffs = [p for suffix in ('.tif', '.tiff') if (p := source.with_suffix(suffix)).exists()]
    assert len(tiffs) == 1, source
    with Image.open(tiffs[0]) as tif:
        assert all(abs(d-600) < 0.1 for d in tif.info['dpi'])
    records.append(dict(name=name, source=str(source), tiff=str(tiffs[0]), grid_size=[1000, 1000],
        experimental_pH=[10.5, 13.5], experimental_E_RHE=[-2.0, 2.0],
        sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    old.close()
    new.close()
for start in range(0, len(thumbs), 6):
    sheet = Image.new('RGB', (2040, 1530), 'white')
    for i, im in enumerate(thumbs[start:start+6]):
        sheet.paste(im, ((i%2)*1020, (i//2)*510))
    sheet.save(OUT / f'review_{start//6+1}.png')
(OUT / 'installation_manifest.json').write_text(json.dumps(records, indent=2))
print('PASS: 49 PDFs, unchanged page dimensions; contact sheets generated.')
if '--install' in sys.argv:
    backup = OUT / 'previous_submission'
    backup.mkdir(exist_ok=True)
    for record in records:
        target = SUB / record['name']
        assert not (backup / target.name).exists(), target
        shutil.copy2(target, backup / target.name)
        shutil.copy2(record['source'], target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']
    print('Installed 49 PDFs; previous PDFs backed up outside submission.')
