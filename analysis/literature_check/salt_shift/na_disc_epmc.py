import sys, json, urllib.request, urllib.parse
q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 40
u="https://www.ebi.ac.uk/europepmc/webservices/rest/search?"+urllib.parse.urlencode({"query":q,"format":"json","resultType":"lite","pageSize":n})
d=json.load(urllib.request.urlopen(u,timeout=60))
print("HITS",d.get("hitCount"))
for r in d["resultList"]["result"]:
    print(r.get("pubYear"),"|",r.get("doi"),"|",r.get("pmcid"),"|",(r.get("authorString") or "")[:40],"|",r.get("title")[:150],"|",r.get("journalTitle"))
