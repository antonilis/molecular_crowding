"""Fetch abstracts from OpenAlex (inverted index) for DOIs: python abstracts.py DOI..."""
import sys, json, urllib.request, urllib.parse
H={'User-Agent':'lit-check (mailto:antoni@powerplug.ai)'}
for d in sys.argv[1:]:
    try:
        j=json.load(urllib.request.urlopen(urllib.request.Request('https://api.openalex.org/works/doi:'+d,headers=H),timeout=30))
        inv=j.get('abstract_inverted_index') or {}
        w=sorted(((p,k) for k,ps in inv.items() for p in ps))
        print('##',d,'|',j.get('title'),'| OA:',j.get('open_access',{}).get('oa_url'))
        print(' '.join(k for p,k in w) or '(no abstract)'); print()
    except Exception as e: print('FAIL',d,e)
