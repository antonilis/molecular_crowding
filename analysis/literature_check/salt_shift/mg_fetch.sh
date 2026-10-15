#!/bin/bash
# mg_: fetch PMC full text (NCBI efetch; fallback PMC HTML). usage: mg_fetch.sh Name:PMCID ...
P=/media/dysk/IChF/stała_równowagi_w_zatłoczeniu_pegami/.venv/bin/python
D=/media/dysk/IChF/stała_równowagi_w_zatłoczeniu_pegami/analysis/literature_check/salt_shift
for x in "$@"; do n=${x%%:*}; id=${x##*:}; out=$D/mg_ft/$n.txt
  $P $D/mg_epmc.py ncbi $id $out >/dev/null 2>&1; sz=$(wc -c <$out 2>/dev/null || echo 0)
  if [ "$sz" -lt 9000 ]; then
    mkdir -p /tmp/mg_dl; curl -sL -A "Mozilla/5.0" "https://pmc.ncbi.nlm.nih.gov/articles/$id/" -o /tmp/mg_dl/$id.html
    $P -I -c "
import re,html,sys
t=open(sys.argv[1],errors='ignore').read()
t=re.sub(r'<script.*?</script>|<style.*?</style>','',t,flags=re.S)
t=html.unescape(re.sub(r'<[^>]+>',' ',t)); t=re.sub(r'\s+',' ',t)
open(sys.argv[2],'w').write(t)" /tmp/mg_dl/$id.html $out
  fi
  echo "$n $id $(wc -c <$out)"; sleep 0.4
done
