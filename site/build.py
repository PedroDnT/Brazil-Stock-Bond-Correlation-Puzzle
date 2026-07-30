"""Inline the woff2 faces into template.html and write index.html.

Run from this directory:  python3 build.py
Edit template.html, never index.html — this script overwrites it.
"""
import base64
import pathlib

SRC = pathlib.Path('template.html')
OUT = pathlib.Path('index.html')
FONTS = {
    '__BODONI__': 'fonts/bodoni.woff2',
    '__ARCHIVO__': 'fonts/archivo.woff2',
    '__PLEX400__': 'fonts/plexmono.woff2',
    '__PLEX500__': 'fonts/plexmono500.woff2',
}

tpl = SRC.read_text()
for placeholder, path in FONTS.items():
    if placeholder not in tpl:
        raise SystemExit(f'{placeholder} missing from {SRC} — font would not load')
    tpl = tpl.replace(placeholder, base64.b64encode(pathlib.Path(path).read_bytes()).decode())

OUT.write_text(tpl)
print(f'wrote {OUT} ({len(tpl):,} bytes)')
