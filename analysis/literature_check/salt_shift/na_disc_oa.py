import sys,json,urllib.request
for doi in sys.argv[1:]:
    try:
        d=json.load(urllib.request.urlopen(f"https://api.openalex.org/works/doi:{doi}?mailto=antoni@powerplug.ai",timeout=60))
    except Exception as e: print(doi,"ERR",e); continue
    ii=d.get("abstract_inverted_index")
    ab=""
    if ii:
        pos={}
        for w,ps in ii.items():
            for p in ps: pos[p]=w
        ab=" ".join(pos[i] for i in sorted(pos))
    print("##",doi,d.get("title"),"| OA:",d.get("open_access",{}).get("oa_url"),"| cited",d.get("cited_by_count")); print(ab or "NOABS"); print()
