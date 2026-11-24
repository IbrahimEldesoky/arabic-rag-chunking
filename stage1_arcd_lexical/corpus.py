import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import json, re, collections
def build():
    rows=json.load(open(P_ARCD))
    docs=collections.OrderedDict()   # title -> list of unique paragraphs (order of appearance)
    for r in rows:
        ps=docs.setdefault(r['title'],[])
        if r['context'] not in ps: ps.append(r['context'])
    titles=list(docs); tid={t:i for i,t in enumerate(titles)}
    doc_text=[]; para_off=[]   # para_off[doc][para_idx] = (start,end) char offsets in doc text
    for t in titles:
        txt=''; offs=[]
        for p in docs[t]:
            if txt: txt+='\n\n'
            offs.append((len(txt),len(txt)+len(p))); txt+=p
        doc_text.append(txt); para_off.append(offs)
    qs=[]
    for r in rows:
        d=tid[r['title']]; pi=docs[r['title']].index(r['context']); ps=para_off[d][pi][0]
        if r['context'][r['start']:r['start']+len(r['ans'])]!=r['ans'] or not r['ans'].strip(): continue
        q=re.sub(r'^[\s\-–—•]+','',r['question']).strip()
        qs.append(dict(id=r['id'],doc=d,question=q,ans=r['ans'],a_s=ps+r['start'],a_e=ps+r['start']+len(r['ans']),split=r['split']))
    return titles,doc_text,para_off,qs
if __name__=='__main__':
    import numpy as np
    t,d,po,q=build()
    print('docs',len(d),'questions',len(q))
    w=[len(x.split()) for x in d]; print('words/doc',np.percentile(w,[0,10,25,50,75,90,100]),'total',sum(w))
    aw=[len(x['ans'].split()) for x in q]; print('answer words',np.percentile(aw,[0,25,50,75,90,100]))
    qw=[len(x['question'].split()) for x in q]; print('question words',np.percentile(qw,[0,50,100]))
    pw=[len(d[i][a:b].split()) for i in range(len(d)) for a,b in po[i]]; print('words/para',np.percentile(pw,[0,25,50,75,100]), len(pw))
    print(q[0]); print(q[100]['question'], '->', q[100]['ans'])
    for x in q: assert d[x['doc']][x['a_s']:x['a_e']]==x['ans']
    print('offsets verified')
