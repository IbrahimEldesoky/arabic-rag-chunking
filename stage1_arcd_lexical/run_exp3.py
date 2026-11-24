import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,time,pickle,numpy as np
from run_experiments import *
R3={}; t0=time.time()
for cfg in cfgs:
    ch=chunk_corpus(cfg,docs,po); texts=[docs[d][a:b] for d,a,b in ch]; evaluate.wlen=np.array([len(t.split()) for t in texts])
    A=Index(texts,'norm_stop_stem'); SA=A.scores(queries)
    C=Index(texts,'char4'); SC=C.scores(queries)
    S={'c4_bm25':SC['bm25'],'c4_tfidf':SC['tfidf'],'c4_lsa':SC['lsa'],
       'c4_hybrid':rrf([SC['bm25'],SC['lsa']]),
       'fusion_lex':rrf([SA['bm25'],SC['bm25']]),
       'fusion3':rrf([SA['bm25'],SC['bm25'],SA['lsa']])}
    for r,sc in S.items(): R3[(cfg_name(cfg),'mix',r)]=evaluate(sc,ch,qs)
    print(cfg_name(cfg),' '.join(f"{r}:{R3[(cfg_name(cfg),'mix',r)]['budget256'].mean():.3f}" for r in S),f'[{time.time()-t0:.0f}s]',flush=True)
pickle.dump(R3,open('res_exp3.pkl','wb'))
