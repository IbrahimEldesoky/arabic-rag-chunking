"""Repair TyDi QA answer offsets: relocate answer string to nearest occurrence of the annotated start. Usage: python fix_offsets.py in.json out.json"""
import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import json,re,sys
r=json.load(open(sys.argv[1])); out=[]
for x in r:
    c,a,s=x['context'],x['ans'],x['start']
    if c[s:s+len(a)]!=a:
        occ=[m.start() for m in re.finditer(re.escape(a),c)]; s=min(occ,key=lambda o:abs(o-s))
    y=dict(x); y['start']=s; out.append(y)
json.dump(out,open(sys.argv[2],'w'),ensure_ascii=False)
