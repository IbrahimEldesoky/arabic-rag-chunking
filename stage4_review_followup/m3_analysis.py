import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import gzip,pickle,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from corpus_lib import build_arcd,build_tydi
F={('ARCD','e5'):'m3_arcd_e5.pkl.gz',('ARCD','bge-m3'):'m3_arcd_bgem3.pkl.gz',('TyDi','e5'):'m3_tydi_e5.pkl.gz',('TyDi','bge-m3'):'m3_tydi_bgem3.pkl.gz'}
D={k:pickle.load(gzip.open(f,'rb')) for k,f in F.items()}
Qs={'ARCD':build_arcd(P_ARCD)[3],'TyDi':build_tydi(P_TYDI)[3]}
rng=np.random.default_rng(31)
CF8=['fixed-32','fixed-64','fixed-128','fixed-256','sliding-64-ov25','sentence-64','sentence-128','paragraph']
def pq(ds,m,s,key='budget256',cfgs=CF8):
    d=D[(ds,m)]; R=d['R']; t=d['tag']; return np.mean([R[(c,t,s)][key].astype(float) for c in cfgs],0)
def info(ds):
    docid=np.array([q['doc'] for q in Qs[ds]]); n=docid.max()+1; return n,[np.where(docid==x)[0] for x in range(n)]
def cboot(a,b,ds,B=3000):
    n,mem=info(ds); dd=a-b; sm=np.array([dd[x].sum() for x in mem]); cn=np.array([len(x) for x in mem])
    o=np.array([(lambda i:sm[i].sum()/cn[i].sum())(rng.integers(0,n,n)) for _ in range(B)]); return dd.mean()*100,np.percentile(o,2.5)*100,np.percentile(o,97.5)*100
pd.set_option('display.width',230)
SYS=['bm25_c4','dense','rrf_equal','rrf_d2']+[f'{n}_a{a}' for a in (50,70,85) for n in ['zall','ztop100','ztop20','mmtop100','mmtop20']]
print('== Hit@256w (%) pooled over 8 chunking configs (whole excluded)')
T=pd.DataFrame({(ds,m):{s:pq(ds,m,s).mean()*100 for s in SYS} for (ds,m) in F}); print(T.round(1).to_string())
print('\n== best system per run:'); print({k:(T[k].idxmax(),round(T[k].max(),1)) for k in T})
rows=[]
for (ds,m) in F:
    for a in (70,85):
        z=pq(ds,m,f'zall_a{a}'); rr=pq(ds,m,'rrf_equal'); dn=pq(ds,m,'dense')
        for nm in ['ztop100','ztop20','mmtop100','mmtop20']:
            x=pq(ds,m,f'{nm}_a{a}'); d,lo,hi=cboot(x,z,ds); rows.append(dict(dataset=ds,model=m,alpha=a/100,contrast=f'{nm} - zall',diff=d,lo=lo,hi=hi))
        for nm in ['zall','ztop100','ztop20','mmtop100','mmtop20']:
            x=pq(ds,m,f'{nm}_a{a}'); d,lo,hi=cboot(x,rr,ds); rows.append(dict(dataset=ds,model=m,alpha=a/100,contrast=f'{nm} - RRF equal',diff=d,lo=lo,hi=hi))
C=pd.DataFrame(rows); C.to_csv('m3_contrasts.csv',index=False)
print('\n== normalisation effect (points; 95% cluster-bootstrap CI)')
for a in (70,85):
    print('dense weight',a/100); print(C[(C.alpha==a/100)&C.contrast.str.endswith('- zall')].assign(res=lambda d:d.apply(lambda r:f'{r["diff"]:+.2f} ({r["lo"]:+.2f},{r["hi"]:+.2f})',axis=1)).pivot(index='contrast',columns=['dataset','model'],values='res').to_string())
print('\n== every fusion variant minus equal-weight RRF (points)')
for a in (70,85):
    print('dense weight',a/100); print(C[(C.alpha==a/100)&C.contrast.str.endswith('- RRF equal')].assign(res=lambda d:d.apply(lambda r:f'{r["diff"]:+.2f} ({r["lo"]:+.2f},{r["hi"]:+.2f})',axis=1)).pivot(index='contrast',columns=['dataset','model'],values='res').to_string())
# worst-case regret per normalisation family, using best alpha chosen per family per run
print('\n== worst-case regret (points) of each fusion family at alpha 0.70 / 0.85, relative to the best of the 19 systems per run')
reg={}
for k in T:
    b=T[k].max()
    for s in ['bm25_c4','dense','rrf_equal','zall_a70','zall_a85','ztop100_a70','ztop100_a85','mmtop100_a70','mmtop100_a85','ztop20_a85']: reg.setdefault(s,[]).append(b-T[k][s])
print(pd.DataFrame({s:{'max regret':max(v),'mean regret':np.mean(v)} for s,v in reg.items()}).T.round(2).to_string())
