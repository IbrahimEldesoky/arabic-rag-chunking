import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,gzip,pickle,numpy as np,pandas as pd
from corpus_lib import build_arcd,build_tydi
ds,pk,tag,out=sys.argv[1:5]
titles,docs,po,qs=(build_arcd(P_ARCD) if ds=='arcd' else build_tydi(P_TYDI)); Q=len(qs)
d=pickle.load(gzip.open(pk,'rb')); R=d['R']; cfgs=sorted({k[0] for k in R}); rng=np.random.default_rng(11)
M=lambda c,s,k: R[(c,tag,s)][k].astype(np.float64)
NOWH=[c for c in cfgs if c!='whole']
docid=np.array([q['doc'] for q in qs]); D=docid.max()+1; members=[np.where(docid==x)[0] for x in range(D)]; cnt=np.array([len(m) for m in members])
def cboot(a,b,B=3000):
    dd=a-b; sums=np.array([dd[m].sum() for m in members]); o=np.array([(lambda i:sums[i].sum()/cnt[i].sum())(rng.integers(0,D,D)) for _ in range(B)])
    return float(dd.mean()),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))
rows=[]
def add(group,name,a,b,metric):
    m,lo,hi=cboot(a,b); rows.append(dict(group=group,contrast=name,metric=metric,diff=m,lo=lo,hi=hi,sig=(lo>0 or hi<0)))
for metric in ['budget256','mrr@10']:
    pool=lambda s: np.mean([M(c,s,metric) for c in NOWH],0)
    for a,b in [('dense','bm25_c4'),('dense','bm25_stem'),('rrf_c4_dense','dense'),('rrf_c4_dense','bm25_c4'),('rrf_stem_dense','dense'),('rrf_c4_dense_w2','dense'),('rrf_stem_c4_dense','dense'),('bm25_c4','bm25_stem')]: add('retriever (pooled 21 cfgs)',f'{a} - {b}',pool(a),pool(b),metric)
    for s in ['dense','bm25_c4']:
        g=lambda c: M(c,s,metric)
        for a,b in [('fixed-32','fixed-16'),('fixed-64','fixed-32'),('fixed-128','fixed-64'),('fixed-256','fixed-128'),('whole','fixed-64'),('whole','fixed-128')]: add(f'size [{s}]',f'{a} - {b}',g(a),g(b),metric)
        for z in (32,64,128):
            add(f'overlap [{s}]',f'sliding-{z}-ov25 - fixed-{z}',g(f'sliding-{z}-ov25'),g(f'fixed-{z}'),metric)
            add(f'boundary [{s}]',f'sentence-{z} - fixed-{z}',g(f'sentence-{z}'),g(f'fixed-{z}'),metric)
C=pd.DataFrame(rows); C.to_csv(out+'_contrasts.csv',index=False)
# best-system ranking (budget256), Holm
sysd={(c,s):M(c,s,'budget256') for c in cfgs for s in ['dense','bm25_c4','rrf_c4_dense','rrf_stem_dense']}
keys=list(sysd); means=np.array([sysd[k].mean() for k in keys]); best=keys[int(means.argmax())]; rk=[]
for k in keys:
    if k==best: continue
    m,lo,hi=cboot(sysd[best],sysd[k],B=2000); p=max(2*min(0,0),0)
    # bootstrap p via re-sampling difference
    dd=sysd[best]-sysd[k]; sums=np.array([dd[mm].sum() for mm in members]); o=np.array([(lambda i:sums[i].sum()/cnt[i].sum())(rng.integers(0,D,D)) for _ in range(2000)])
    p=max(2*min((o<=0).mean(),(o>=0).mean()),1/2000); rk.append(dict(system=f'{k[0]} | {k[1]}',mean=float(sysd[k].mean()),diff_vs_best=-m,lo=-hi,hi=-lo,p=p))
RK=pd.DataFrame(rk).sort_values('mean',ascending=False); n=len(RK); order=np.argsort(RK.p.values); adj=np.empty(n); run=0
for i,j in enumerate(order): run=max(run,(n-i)*RK.p.values[j]); adj[j]=min(1,run)
RK['p_holm']=adj; RK=pd.concat([pd.DataFrame([dict(system=f'{best[0]} | {best[1]} (BEST)',mean=float(sysd[best].mean()),diff_vs_best=0,lo=0,hi=0,p=np.nan,p_holm=np.nan)]),RK]); RK.to_csv(out+'_ranking.csv',index=False)
# variance decomposition: chunking cfg x system
def decomp(Y,B=1000):
    def sh(Mc):
        g=Mc.mean(); mc=Mc.mean(1); mr=Mc.mean(0); C_,R_=Mc.shape; ssc=R_*((mc-g)**2).sum(); ssr=C_*((mr-g)**2).sum(); sst=((Mc-g)**2).sum(); return np.array([ssc,ssr,sst-ssc-ssr])/sst
    pt=sh(Y.mean(0)); bs=np.array([sh(Y[rng.integers(0,Q,Q)].mean(0)) for _ in range(B)]); return pt,np.percentile(bs,2.5,0),np.percentile(bs,97.5,0)
restricted=['fixed-64','fixed-128','fixed-256','sliding-64-ov25','sliding-64-ov50','sliding-128-ov25','sliding-128-ov50','sliding-256-ov25','sliding-256-ov50','sentence-64','sentence-128','sentence-256','paragraph']
vd=[]
for lab,cs in [('all 21',NOWH),('>=64 words',restricted)]:
    for sysset,sl in [('bm25_c4+dense',['bm25_c4','dense']),('bm25_stem,bm25_c4,dense,rrf_c4_dense',['bm25_stem','bm25_c4','dense','rrf_c4_dense'])]:
        for key in ['mrr@10','budget256','budget512']:
            Y=np.stack([np.stack([M(c,s,key) for s in sl],1) for c in cs],1); pt,lo,hi=decomp(Y,500)
            vd.append(dict(design=lab,systems=sysset,metric=key,chunking=pt[0],chunk_lo=lo[0],chunk_hi=hi[0],retriever=pt[1],ret_lo=lo[1],ret_hi=hi[1],interaction=pt[2],chunk_range=Y.mean((0,2)).max()-Y.mean((0,2)).min(),retr_range=Y.mean((0,1)).max()-Y.mean((0,1)).min()))
pd.DataFrame(vd).to_csv(out+'_variance.csv',index=False)
# per-config table
T=[]
for c in cfgs:
    for s in ['bm25_c4','dense','rrf_c4_dense']:
        T.append(dict(config=c,system=s,**{k:float(M(c,s,k).mean()) for k in ['hit@1','mrr@10','budget128','budget256','budget512','doc_hit@1']}))
pd.DataFrame(T).to_csv(out+'_table.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(C[C.metric=='budget256'].round(4).to_string(index=False)); print(); print(RK.head(8).round(4).to_string(index=False)); print(); print(pd.DataFrame(vd).round(3).to_string(index=False))
