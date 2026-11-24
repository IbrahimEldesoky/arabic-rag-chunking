import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import gzip,pickle,numpy as np,pandas as pd
from corpus_lib import build_arcd,build_tydi
F={('ARCD','e5'):'arcd_e5base_v2.pkl.gz',('ARCD','bge-m3'):'arcd_bgem3.pkl.gz',('TyDi','e5'):'tydi_e5base_v2.pkl.gz',('TyDi','bge-m3'):'tydi_bgem3.pkl.gz'}
D={}; 
for k,f in F.items(): d=pickle.load(gzip.open(f,'rb')); D[k]=(d['R'],d['tag'])
Qs={'ARCD':build_arcd(P_ARCD)[3],'TyDi':build_tydi(P_TYDI)[3]}
# reproducibility: e5 v2 vs v1 dense
v1=pickle.load(gzip.open('arcd_e5base.pkl.gz','rb'))['R']; R,t=D[('ARCD','e5')]
print('e5 v2 vs v1 dense per-question max diff (ARCD):',max(np.abs(R[(c,t,'dense')]['budget256']-v1[(c,t,'dense')]['budget256']).max() for c in sorted({k[0] for k in R})))
SYS=['bm25_stem','bm25_c4','dense','rrf_c4_dense','rrf_c4_dense_w2','rrf_c4_dense_d2','rrf_c4_dense_d4','z_c4_dense_a30','z_c4_dense_a50','z_c4_dense_a70','z_c4_dense_a85','rrf_stem_dense','rrf_stem_c4_dense']
pd.set_option('display.width',220)
pool={}
for (ds,m),(R,t) in D.items():
    cf=[c for c in sorted({k[0] for k in R}) if c!='whole']
    for s in SYS: pool[(ds,m,s)]=float(np.mean([R[(c,t,s)]['budget256'].mean() for c in cf]))
P=pd.Series(pool).unstack([0,1]).loc[SYS]; print((P*100).round(1).to_string())
# regret per model (best among the 13 systems on that dataset/model), worst-case across datasets
reg={}
for m in ['e5','bge-m3']:
    for ds in ['ARCD','TyDi']:
        b=P[(ds,m)].max(); 
        for s in SYS: reg[(m,ds,s)]=b-P[(ds,m)][s]
Rg=pd.Series(reg).unstack([0,1]).loc[SYS]; Rg[('e5','worst')]=Rg['e5'].max(1); Rg[('bge-m3','worst')]=Rg['bge-m3'].max(1)
print('\nregret (points)'); print((Rg*100).round(1).to_string())
# best system per dataset/model
for k in P.columns: print(k,'best:',P[k].idxmax(),round(P[k].max()*100,1))
