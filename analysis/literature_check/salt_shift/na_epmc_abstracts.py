"""Fetch Europe PMC core records (abstract, OA flags, PMCID) and Crossref metadata for a list of DOIs.
Usage: python -I na_epmc_abstracts.py DOI [DOI ...]   -> prints, and appends JSON lines to na_epmc_abstracts.jsonl"""
import sys, json, os, urllib.request, urllib.parse
H = {'User-Agent': 'lit-check (mailto:antoni@powerplug.ai)'}
HERE = os.path.dirname(os.path.abspath(__file__))
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))
out = open(os.path.join(HERE, 'na_epmc_abstracts.jsonl'), 'a')
for doi in sys.argv[1:]:
    rec = {'doi': doi}
    try:
        m = get('https://api.crossref.org/works/' + urllib.parse.quote(doi))['message']
        rec['crossref'] = dict(title=m.get('title', [''])[0], journal=(m.get('container-title') or [''])[0],
                               volume=m.get('volume'), page=m.get('page'),
                               year=(m.get('issued', {}).get('date-parts') or [[None]])[0][0],
                               authors=[f"{a.get('family','')} {a.get('given','')}" for a in m.get('author', [])],
                               license=[l.get('URL') for l in m.get('license', [])])
    except Exception as e:
        rec['crossref'] = 'FAIL ' + str(e)
    try:
        r = get('https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&query='
                + urllib.parse.quote(f'DOI:"{doi}"'))['resultList']['result']
        if r:
            r = r[0]
            rec['epmc'] = dict(pmid=r.get('pmid'), pmcid=r.get('pmcid'), oa=r.get('isOpenAccess'),
                               abstract=r.get('abstractText'))
    except Exception as e:
        rec['epmc'] = 'FAIL ' + str(e)
    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    c = rec['crossref']
    print('=' * 80)
    print(doi, '|', c if isinstance(c, str) else f"{c['title']} | {c['journal']} {c['volume']}, {c['page']} ({c['year']}) | {'; '.join(c['authors'])}")
    e = rec.get('epmc')
    if isinstance(e, dict):
        print('PMCID', e['pmcid'], 'OA', e['oa'])
        print('ABS:', e['abstract'])
