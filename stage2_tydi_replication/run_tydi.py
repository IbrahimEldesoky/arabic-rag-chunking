import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,time,pickle,numpy as np
from corpus import build; from chunkers import *; from retrieval import *
titles,docs,po,qs=build(); queries=[q['question'] for q in qs]; Qn=len(qs); BS=250
cfgs=make_configs(); only=sys.argv[1:] 
R={}; R3={}; t0=time.time()
def cat(parts):
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
for cfg in cfgs:
    nm=cfg_name(cfg)
    if only and nm not in only: continue
    ch=chunk_corpus(cfg,docs,po); texts=[docs[d][a:b] for d,a,b in ch]; evaluate.wlen=np.array([len(t.split()) for t in texts])
    IX={'norm_stop':Index(texts,'norm_stop'),'norm_stop_stem':Index(texts,'norm_stop_stem'),'char4':Index(texts,'char4')}
    for m in ('raw','norm','norm_stem'): IX[m]=Index(texts,m,lsa_dim=0)
    acc={}
    for s in range(0,Qn,BS):
        qb=queries[s:s+BS]; qsb=qs[s:s+BS]; SC={m:IX[m].scores(qb) for m in IX}
        for m in ('norm_stop','norm_stop_stem'):
            S=SC[m]; S['hybrid']=rrf([S['bm25'],S['lsa']])
        C=SC['char4']; A=SC['norm_stop_stem']
        sets={}
        for m in ('norm_stop','norm_stop_stem'):
            for r in ('bm25','tfidf','lsa','hybrid'): sets[(m,r)]=SC[m][r]
        for m in ('raw','norm','norm_stem','char4'): sets[(m,'bm25')]=SC[m]['bm25']
        sets[('mix','c4_bm25')]=C['bm25']; sets[('mix','c4_tfidf')]=C['tfidf']; sets[('mix','c4_lsa')]=C['lsa']
        sets[('mix','c4_hybrid')]=rrf([C['bm25'],C['lsa']]); sets[('mix','fusion_lex')]=rrf([A['bm25'],C['bm25']]); sets[('mix','fusion3')]=rrf([A['bm25'],C['bm25'],A['lsa']])
        for k,S in sets.items(): acc.setdefault(k,[]).append(evaluate(S,ch,qsb))
        del SC,sets
    for (m,r),parts in acc.items():
        (R3 if m=='mix' else R)[(nm,m,r)]=cat(parts)
    print(f'{nm:18s} chunks={len(ch):6d} bm25stem MRR={R[(nm,"norm_stop_stem","bm25")]["mrr@10"].mean():.3f} B256={R[(nm,"norm_stop_stem","bm25")]["budget256"].mean():.3f} [{time.time()-t0:.0f}s]',flush=True)
    pickle.dump(R,open('res_all.pkl','wb')); pickle.dump(R3,open('res_exp3.pkl','wb'))
