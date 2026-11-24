import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,time,pickle,itertools,numpy as np,pandas as pd
from corpus import build; from chunkers import *; from retrieval import *
titles,docs,po,qs=build(); queries=[q['question'] for q in qs]
CF=[('fixed',64,0.0),('sentence',128,0.0),('sliding',64,0.25)]; MODE='norm_stop_stem'; out=[]
def m(S,ch): 
    r=evaluate(S,ch,qs); return round(float(r['budget256'].mean()),4),round(float(r['mrr@10'].mean()),4)
for cfg in CF:
    ch=chunk_corpus(cfg,docs,po); texts=[docs[d][a:b] for d,a,b in ch]; evaluate.wlen=np.array([len(t.split()) for t in texts]); nm=cfg_name(cfg)
    # BM25 k1 x b
    for k1,b in itertools.product([0.6,0.9,1.2,1.5,2.0],[0.3,0.5,0.75,0.9,1.0]):
        S=Index(texts,MODE,k1=k1,b=b,lsa_dim=0).scores(queries)['bm25']; h,mr=m(S,ch); out.append(dict(cfg=nm,exp='bm25_k1_b',k1=k1,b=b,budget256=h,mrr=mr))
    # LSA dim + hybrid
    for dim in [128,256,512,800]:
        I=Index(texts,MODE,lsa_dim=dim); S=I.scores(queries)
        h,mr=m(S['lsa'],ch); out.append(dict(cfg=nm,exp='lsa_dim',dim=dim,system='lsa',budget256=h,mrr=mr))
        h,mr=m(rrf([S['bm25'],S['lsa']]),ch); out.append(dict(cfg=nm,exp='lsa_dim',dim=dim,system='hybrid',budget256=h,mrr=mr))
    # RRF k and weights (dim 128)
    I=Index(texts,MODE,lsa_dim=512); S=I.scores(queries)
    for k,w in itertools.product([10,60,200],[1,2,4,8]):
        h,mr=m(rrf([S['bm25'],S['lsa']],k=k,weights=[w,1.0]),ch); out.append(dict(cfg=nm,exp='rrf',k=k,w_bm25=w,budget256=h,mrr=mr))
    h,mr=m(S['bm25'],ch); out.append(dict(cfg=nm,exp='ref_bm25',budget256=h,mrr=mr))
    print(nm,'done',flush=True)
pd.DataFrame(out).to_csv('results/sensitivity.csv',index=False)
df=pd.DataFrame(out)
print('--- BM25 k1 x b (mean over 3 cfgs, budget256)'); print(df[df.exp=='bm25_k1_b'].groupby(['k1','b']).budget256.mean().unstack().round(3).to_string())
print('--- LSA dim'); print(df[df.exp=='lsa_dim'].groupby(['system','dim']).budget256.mean().unstack().round(3).to_string())
print('--- RRF k x w_bm25 (hybrid)'); print(df[df.exp=='rrf'].groupby(['k','w_bm25']).budget256.mean().unstack().round(3).to_string())
print('ref bm25',df[df.exp=='ref_bm25'].budget256.mean().round(3))
