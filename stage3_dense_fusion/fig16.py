import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import warnings; warnings.filterwarnings('ignore')
exec(open('round2.py').read().split("# reproducibility")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def pm(ds,m,s):
    R,t=D[(ds,m)]; cf=[c for c in sorted({k[0] for k in R}) if c!='whole']; return float(np.mean([R[(c,t,s)]['budget256'].mean() for c in cf]))*100
al=[0,0.3,0.5,0.7,0.85,1.0]; names=['bm25_c4','z_c4_dense_a30','z_c4_dense_a50','z_c4_dense_a70','z_c4_dense_a85','dense']
fig,ax=plt.subplots(1,2,figsize=(11,3.9)); col={'e5':'#d62728','bge-m3':'#1f77b4'}
for a,ds in zip(ax,['ARCD','TyDi']):
    for m,ls in [('e5','-'),('bge-m3','--')]:
        a.plot(al,[pm(ds,m,s) for s in names],'o'+ls,color=col[m],label=f'score fusion, {m}',ms=4)
        a.plot([0.5],[pm(ds,m,'rrf_c4_dense')],'D',color=col[m],mfc='white',ms=8,label=f'equal-weight RRF, {m}')
    a.set_title(ds+' (pooled over 21 chunking configurations)',fontsize=10); a.set_xlabel('dense weight (0 = char-4-gram BM25 only, 1 = dense only)'); a.grid(alpha=.3); a.spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Hit @ 256-word budget (%)'); ax[1].legend(fontsize=7.5,loc='lower right'); plt.tight_layout(); plt.savefig('fig16_fusion.png',dpi=140); print('ok')
