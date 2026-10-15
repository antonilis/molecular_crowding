"""Crossref helper: python crossref.py doi <DOI>... | python crossref.py q '<bibliographic query>'"""
import sys, json, urllib.request, urllib.parse
H={'User-Agent':'lit-check (mailto:antoni@powerplug.ai)'}
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
def fmt(m):
    a=', '.join(f"{x.get('family','')} {x.get('given','')[:1]}." for x in m.get('author',[]))
    y=(m.get('issued',{}).get('date-parts') or [[None]])[0][0]
    return f"{a} | {m.get('title',[''])[0]} | {(m.get('container-title') or [''])[0]} {m.get('volume','')}, {m.get('page','')} ({y}) | DOI {m.get('DOI')}"
if sys.argv[1]=='doi':
    for d in sys.argv[2:]:
        try: print('OK ',fmt(get('https://api.crossref.org/works/'+urllib.parse.quote(d))['message']))
        except Exception as e: print('FAIL',d,e)
else:
    for q in sys.argv[2:]:
        print('##',q)
        r=get('https://api.crossref.org/works?rows=5&query.bibliographic='+urllib.parse.quote(q))
        for m in r['message']['items']: print('  ',fmt(m))
