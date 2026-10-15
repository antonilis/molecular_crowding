"""mg_: OpenAlex search: python mg_oa.py '<search terms>' [n]"""
import sys, json, urllib.request, urllib.parse
H={'User-Agent':'lit-check (mailto:antoni@powerplug.ai)'}
n=sys.argv[2] if len(sys.argv)>2 else '15'
u='https://api.openalex.org/works?per-page='+n+'&search='+urllib.parse.quote(sys.argv[1])
j=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=60))
print('hits',j['meta']['count'])
for w in j['results']:
    print(f"- {w.get('publication_year')} | {(w.get('doi') or '').replace('https://doi.org/','')} | cit={w.get('cited_by_count')} | {w.get('title')} | {((w.get('primary_location') or {}).get('source') or {}).get('display_name')} | OA={w.get('open_access',{}).get('oa_url')}")
