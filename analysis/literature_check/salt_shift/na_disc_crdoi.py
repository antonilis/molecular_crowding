import sys,json,urllib.request,urllib.parse
for doi in sys.argv[1:]:
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/"+urllib.parse.quote(doi),headers={"User-Agent":"mailto:antoni@powerplug.ai"}),timeout=60))["message"]
        a="; ".join((x.get("family","")+" "+x.get("given","")) for x in r.get("author",[]))
        print(doi,"|",a,"|",r["title"][0],"|",r.get("container-title",[""])[0],r.get("volume"),r.get("issue"),r.get("page"),r["issued"]["date-parts"][0][0])
    except Exception as e: print(doi,"ERR",e)
