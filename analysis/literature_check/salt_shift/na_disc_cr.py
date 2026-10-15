import sys, json, urllib.request, urllib.parse
q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 5
u="https://api.crossref.org/works?"+urllib.parse.urlencode({"query.bibliographic":q,"rows":n,"select":"DOI,title,author,container-title,volume,page,issued"})
d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"mailto:antoni@powerplug.ai"}),timeout=60))
for r in d["message"]["items"]:
    a="; ".join((x.get("family","")+" "+x.get("given","")[:1]) for x in r.get("author",[])[:6])
    print(r["issued"]["date-parts"][0][0],"|",r["DOI"],"|",a,"|",(r.get("title") or [""])[0][:140],"|",(r.get("container-title") or [""])[0],r.get("volume"),r.get("page"))
