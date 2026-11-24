import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from corpus_lib import build_arcd,build_tydi
from chunkers import *; from retrieval import *
BUD=(32,64,96,128,192,256,384,512,768,1024)
CF=[('fixed',32,0.0),('fixed',64,0.0),('fixed',128,0.0),('fixed',256,0.0),('sliding',64,0.25),('sentence',64,0.0),('sentence',128,0.0),('paragraph',0,0.0),('whole',0,0.0)]
rows=[]
for ds,b,f in [('ARCD',build_arcd,P_ARCD),('TyDi',build_tydi,P_TYDI)]:
    t,docs,po,qs=b(f); queries=[q['question'] for q in qs]
    for cfg in CF:
        ch=chunk_corpus(cfg,docs,po); texts=[docs[d][a:e] for d,a,e in ch]; evaluate.wlen=np.array([len(x.split()) for x in texts])
        S=Index(texts,'char4',lsa_dim=0).scores(queries)['bm25']
        # evaluate in blocks to bound memory
        acc={}
        for s in range(0,len(qs),250):
            r=evaluate(S[s:s+250],ch,qs[s:s+250],budgets=BUD)
            for B in BUD: acc.setdefault(B,[]).append(r[f'budget{B}'])
        for B in BUD: rows.append(dict(dataset=ds,config=cfg_name(cfg),budget=B,hit=float(np.concatenate(acc[B]).mean())))
    print(ds,'done',flush=True)
M=pd.DataFrame(rows); M.to_csv('m2_budget_curve_bm25.csv',index=False)
pd.set_option('display.width',220)
for ds in ['ARCD','TyDi']:
    t=M[M.dataset==ds].pivot(index='config',columns='budget',values='hit').loc[[cfg_name(c) for c in CF]]*100
    print('\n',ds,'- BM25 (char 4-grams), Hit@Budget % by budget (words)'); print(t.round(1).to_string())
    print('best config per budget:',{int(B):t[B].idxmax() for B in BUD})
fig,ax=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
for a,ds in zip(ax,['ARCD','TyDi']):
    for c in [cfg_name(x) for x in CF]:
        d=M[(M.dataset==ds)&(M.config==c)]; a.plot(d.budget,d.hit*100,'o-',ms=3,label=c)
    a.set_xscale('log',base=2); a.set_xticks(BUD); a.set_xticklabels(BUD,rotation=45,fontsize=8); a.set_title(ds+' (char-4-gram BM25)',fontsize=10); a.set_xlabel('context budget (words)'); a.grid(alpha=.3); a.spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Hit@Budget (%)'); ax[1].legend(fontsize=7,ncol=2,loc='lower right'); plt.tight_layout(); plt.savefig('m2_budget_curve.png',dpi=140)
