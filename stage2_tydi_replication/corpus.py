import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import json,re
def build():
    rows=json.load(open(P_TYDI))
    docs=[]; idx={}
    for r in rows:
        if r['context'] not in idx: idx[r['context']]=len(docs); docs.append(r['context'])
    para_off=[[(0,len(d))] for d in docs]; titles=[str(i) for i in range(len(docs))]
    qs=[]
    for r in rows:
        d=idx[r['context']]; q=re.sub(r'^[\s\-–—•]+','',r['question']).strip()
        qs.append(dict(id=r['id'],doc=d,question=q,ans=r['ans'],a_s=r['start'],a_e=r['start']+len(r['ans']),split=r['split']))
    return titles,docs,para_off,qs
