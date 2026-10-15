import sys, json, urllib.request, urllib.parse
for doi in sys.argv[1:]:
    u="https://www.ebi.ac.uk/europepmc/webservices/rest/search?"+urllib.parse.urlencode({"query":f'DOI:"{doi}"',"format":"json","resultType":"core"})
    d=json.load(urllib.request.urlopen(u,timeout=60))["resultList"]["result"]
    for r in d[:1]:
        print("##",doi,r.get("pmcid"),r.get("isOpenAccess"),"|",r.get("title")); print(r.get("abstractText","NOABS")); print()
