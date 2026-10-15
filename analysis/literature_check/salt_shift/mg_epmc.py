"""mg_: Europe PMC helper.
  python mg_epmc.py s '<query>' [n]      -> search (core), print DOI|PMCID|title|year|journal|isOA
  python mg_epmc.py ft PMCID outfile     -> download full-text XML from Europe PMC, save plain text
  python mg_epmc.py abs DOI              -> print abstract
"""
import sys, json, re, urllib.request, urllib.parse
H={'User-Agent':'lit-check (mailto:antoni@powerplug.ai)'}
def get(u,raw=False):
    r=urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=60).read()
    return r if raw else json.loads(r)
B='https://www.ebi.ac.uk/europepmc/webservices/rest/'
c=sys.argv[1]
if c=='s':
    n=sys.argv[3] if len(sys.argv)>3 else '25'
    j=get(B+'search?format=json&resultType=core&pageSize='+n+'&query='+urllib.parse.quote(sys.argv[2]))
    print('hits',j.get('hitCount'))
    for r in j['resultList']['result']:
        print(f"- {r.get('pubYear')} | {r.get('doi')} | {r.get('pmcid')} | {r.get('authorString','')[:60]} | {r.get('title')} | {r.get('journalTitle') or r.get('journalInfo',{}).get('journal',{}).get('title')} | OA={r.get('isOpenAccess')}")
elif c=='ft':
    x=get(B+sys.argv[2]+'/fullTextXML',raw=True).decode()
    t=re.sub(r'<[^>]+>',' ',x); t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t)
    open(sys.argv[3],'w').write(t); print(len(t),'chars ->',sys.argv[3])
elif c=='abs':
    for d in sys.argv[2:]:
        j=get(B+'search?format=json&resultType=core&query=DOI:'+urllib.parse.quote('"'+d+'"'))
        for r in j['resultList']['result']:
            print('##',d,r.get('pmcid'),r.get('title')); print(r.get('abstractText')); print()
if c=='ncbi':  # python mg_epmc.py ncbi PMCID outfile  (NCBI efetch PMC XML)
    u='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id='+sys.argv[2].replace('PMC','')
    x=get(u,raw=True).decode()
    t=re.sub(r'<[^>]+>',' ',x); t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n',t)
    open(sys.argv[3],'w').write(t); print(len(t),'chars ->',sys.argv[3])
