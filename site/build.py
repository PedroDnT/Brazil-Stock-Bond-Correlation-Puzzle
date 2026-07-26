import base64,pathlib
tpl=pathlib.Path('template.html').read_text()
m={'__BODONI__':'fonts/bodoni.woff2','__ARCHIVO__':'fonts/archivo.woff2',
   '__PLEX400__':'fonts/plexmono.woff2','__PLEX500__':'fonts/plexmono500.woff2'}
for k,v in m.items():
    tpl=tpl.replace(k, base64.b64encode(pathlib.Path(v).read_bytes()).decode())
pathlib.Path('brazil-correlation.html').write_text(tpl)
print('bytes:',len(tpl))
