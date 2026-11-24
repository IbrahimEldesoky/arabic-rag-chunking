import re
WORD=re.compile(r'\S+')
SENT=re.compile(r'[^.!?؟؛\n]+[.!?؟؛]*')
def words(doc): return [(m.start(),m.end()) for m in WORD.finditer(doc)]
def fixed(doc,size,overlap=0.0):
    w=words(doc); step=max(1,int(round(size*(1-overlap)))); out=[]
    for i in range(0,len(w),step):
        seg=w[i:i+size]; out.append((seg[0][0],seg[-1][1]))
        if i+size>=len(w): break
    return out
def sentence(doc,maxw):
    sents=[(m.start(),m.end()) for m in SENT.finditer(doc) if m.group().strip()]
    # trim whitespace
    s2=[]
    for a,b in sents:
        t=doc[a:b]; a+=len(t)-len(t.lstrip()); b-=len(t)-len(t.rstrip()); 
        if b>a: s2.append((a,b))
    out=[]; cur=None; cw=0
    def nw(a,b): return len(doc[a:b].split())
    for a,b in s2:
        n=nw(a,b)
        if n>maxw:                                   # very long sentence -> flush and split by words
            if cur: out.append(cur); cur=None; cw=0
            out+= [(a+x,a+y) for x,y in fixed(doc[a:b],maxw,0)]
            continue
        if cur and cw+n>maxw: out.append(cur); cur=None; cw=0
        if cur: cur=(cur[0],b); cw+=n
        else: cur=(a,b); cw=n
    if cur: out.append(cur)
    return out
def paragraph(doc,para_offs): return list(para_offs)
def whole(doc): return [(0,len(doc))]
def make_configs(sizes=(16,32,64,128,256)):
    cfgs=[]
    for s in sizes: cfgs.append(('fixed',s,0.0))
    for s in sizes:
        for ov in (0.25,0.5): cfgs.append(('sliding',s,ov))
    for s in sizes: cfgs.append(('sentence',s,0.0))
    cfgs+= [('paragraph',0,0.0),('whole',0,0.0)]
    return cfgs
def chunk_corpus(cfg,docs,para_offs):
    kind,s,ov=cfg; chunks=[]   # (doc_id,start,end)
    for d,txt in enumerate(docs):
        if kind=='fixed': sp=fixed(txt,s,0.0)
        elif kind=='sliding': sp=fixed(txt,s,ov)
        elif kind=='sentence': sp=sentence(txt,s)
        elif kind=='paragraph': sp=paragraph(txt,para_offs[d])
        else: sp=whole(txt)
        chunks+= [(d,a,b) for a,b in sp]
    return chunks
def cfg_name(cfg):
    k,s,o=cfg
    if k=='sliding': return f'sliding-{s}-ov{int(o*100)}'
    if k in('paragraph','whole'): return k
    return f'{k}-{s}'
