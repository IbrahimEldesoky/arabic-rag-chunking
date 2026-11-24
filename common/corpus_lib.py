import json,re
def _qs(rows,docidx):
    qs=[]
    for r in rows:
        d,off=docidx(r); q=re.sub(r'^[\s\-–—•]+','',r['question']).strip()
        if r['context'][r['start']:r['start']+len(r['ans'])]!=r['ans'] or not r['ans'].strip(): continue
        qs.append(dict(id=r['id'],doc=d,question=q,ans=r['ans'],a_s=off+r['start'],a_e=off+r['start']+len(r['ans'])))
    return qs
def build_arcd(path):
    rows=json.load(open(path)); docs={}
    for r in rows:
        ps=docs.setdefault(r['title'],[])
        if r['context'] not in ps: ps.append(r['context'])
    titles=list(docs); tid={t:i for i,t in enumerate(titles)}; text=[]; offs=[]
    for t in titles:
        txt=''; o=[]
        for p in docs[t]:
            if txt: txt+='\n\n'
            o.append((len(txt),len(txt)+len(p))); txt+=p
        text.append(txt); offs.append(o)
    def di(r):
        d=tid[r['title']]; pi=docs[r['title']].index(r['context']); return d,offs[d][pi][0]
    return titles,text,offs,_qs(rows,di)
def build_tydi(path):
    rows=json.load(open(path)); docs=[]; idx={}
    for r in rows:
        if r['context'] not in idx: idx[r['context']]=len(docs); docs.append(r['context'])
    return [str(i) for i in range(len(docs))],docs,[[(0,len(d))] for d in docs],_qs(rows,lambda r:(idx[r['context']],0))
