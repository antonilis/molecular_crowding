import sys,re
t=open(f"na_disc_ft/{sys.argv[1]}.xml",encoding="utf8",errors="ignore").read()
t=re.sub(r"<[^>]+>"," ",t); t=re.sub(r"\s+"," ",t)
i=t.find(sys.argv[2]); n=int(sys.argv[3]) if len(sys.argv)>3 else 4000
print(t[i:i+n])
