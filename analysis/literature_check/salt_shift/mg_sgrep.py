"""mg_: sentence grep: python mg_sgrep.py file 'regex' [maxlen]"""
import sys,re
t=open(sys.argv[1]).read().replace('\n',' ')
s=re.split(r'(?<=[.;])\s+(?=[A-Z(])',t)
L=int(sys.argv[3]) if len(sys.argv)>3 else 600
for i,x in enumerate(s):
    if re.search(sys.argv[2],x,re.I): print(f'[{i}]',x[:L].strip())
