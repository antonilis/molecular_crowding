import sys, re, urllib.request, os
pmc=sys.argv[1]; pats=sys.argv[2:]
fn=f"na_disc_ft/{pmc}.xml"
if not os.path.exists(fn) or os.path.getsize(fn)<10000:
    try:
        x=urllib.request.urlopen(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML",timeout=90).read()
    except Exception as e:
        x=urllib.request.urlopen(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmc}/unicode",timeout=90).read()
    open(fn,"wb").write(x)
t=open(fn,encoding="utf8",errors="ignore").read()
t=re.sub(r'<infon[^>]*>[^<]*</infon>',' ',t)
t=re.sub(r"<[^>]+>"," ",t); t=re.sub(r"\s+"," ",t)
print("LEN",len(t))
for p in pats:
    for m in re.finditer(p,t,flags=re.I):
        s=max(0,m.start()-250); print("...",t[s:m.end()+250],"\n")
