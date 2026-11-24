import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,time,json,pickle,numpy as np
from corpus import build; from chunkers import *; from retrieval import *
titles,docs,po,qs=build(); queries=[q['question'] for q in qs]
cfgs=make_configs()
RES={}   # (cfgname, mode, retriever) -> metric dict
def run(cfg,mode,rets=('bm25','tfidf','lsa','hybrid'),**kw):
    ch=chunk_corpus(cfg,docs,po); texts=[docs[d][a:b] for d,a,b in ch]
    evaluate.wlen=np.array([len(t.split()) for t in texts])
    idx=Index(texts,mode,**kw); S=idx.scores(queries)
    S['hybrid']=rrf([S['bm25'],S['lsa']],k=kw.get('rrf_k',60))
    return {r:evaluate(S[r],ch,qs) for r in rets},len(ch)
if __name__=='__main__':
    modes=sys.argv[1].split(',') if len(sys.argv)>1 else ['norm_stop']
    t0=time.time()
    for mode in modes:
        for cfg in cfgs:
            rets=('bm25','tfidf','lsa','hybrid') if mode in('norm_stop','norm_stop_stem') else ('bm25',)
            res,n=run(cfg,mode,rets)
            for r,m in res.items(): RES[(cfg_name(cfg),mode,r)]=m
            print(f'{mode:15s} {cfg_name(cfg):18s} n={n:5d} '+' '.join(f"{r}:MRR={res[r]['mrr@10'].mean():.3f}" for r in res),f'[{time.time()-t0:.0f}s]',flush=True)
    pickle.dump(RES,open(f'res_{"_".join(modes)}.pkl','wb'))
