import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,numpy as np,pandas as pd
exec(open('analysis.py').read().split('# ---------- chunk stats')[0])
rng=np.random.default_rng(7)
def boot_idx(B): return rng.integers(0,Q,size=(B,Q))
def decomp(Y,B=1000):
    def sh(Mc):
        g=Mc.mean(); mc=Mc.mean(1); mr=Mc.mean(0); C,Rr=Mc.shape
        ssc=Rr*((mc-g)**2).sum(); ssr=C*((mr-g)**2).sum(); sst=((Mc-g)**2).sum(); return np.array([ssc,ssr,sst-ssc-ssr])/sst
    pt=sh(Y.mean(0)); idx=boot_idx(B); bs=np.array([sh(Y[i].mean(0)) for i in idx]); return pt,np.percentile(bs,2.5,0),np.percentile(bs,97.5,0)
RET=['bm25','tfidf','lsa','hybrid']
restricted=['fixed-64','fixed-128','fixed-256','sliding-64-ov25','sliding-64-ov50','sliding-128-ov25','sliding-128-ov50','sliding-256-ov25','sliding-256-ov50','sentence-64','sentence-128','sentence-256','paragraph']
rows=[]
for mode in ['norm_stop','norm_stop_stem']:
  for rets,lab in [(RET,'4 retrievers'),(['bm25','tfidf'],'lexical only')]:
    for key,reg in [('mrr@10','MRR@10'),('budget256','Budget-256'),('budget512','Budget-512')]:
        Y=np.stack([np.stack([M(c,mode,r,key) for r in rets],1) for c in restricted],1); pt,lo,hi=decomp(Y)
        rows.append(dict(pipeline=mode,retrievers=lab,regime=reg,chunking_share=pt[0],chunk_lo=lo[0],chunk_hi=hi[0],retriever_share=pt[1],ret_lo=lo[1],ret_hi=hi[1],interaction=pt[2],chunk_range=Y.mean((0,2)).max()-Y.mean((0,2)).min(),retr_range=Y.mean((0,1)).max()-Y.mean((0,1)).min()))
pd.DataFrame(rows).to_csv('results/variance_decomposition_restricted.csv',index=False)
docid=np.array([q['doc'] for q in qs]); D=docid.max()+1; members=[np.where(docid==d)[0] for d in range(D)]; cnt=np.array([len(m) for m in members])
def cboot(a,b,B=3000):
    d=a-b; sums=np.array([d[m].sum() for m in members]); out=np.array([ (lambda i: sums[i].sum()/cnt[i].sum())(rng.integers(0,D,D)) for _ in range(B)]); return d.mean(),np.percentile(out,2.5),np.percentile(out,97.5)
NOWH=[c for c in cfgs if c!='whole']; k='budget256'; st='norm_stop_stem'
g=lambda c,m=st,r='bm25': M(c,m,r,k); pool=lambda m,r: np.mean([M(c,m,r,k) for c in NOWH],0)
T=[('sliding-32-ov25 - fixed-32',g('sliding-32-ov25'),g('fixed-32')),('sliding-64-ov25 - fixed-64',g('sliding-64-ov25'),g('fixed-64')),('sliding-128-ov25 - fixed-128',g('sliding-128-ov25'),g('fixed-128')),
('sentence-32 - fixed-32',g('sentence-32'),g('fixed-32')),('sentence-64 - fixed-64',g('sentence-64'),g('fixed-64')),('sentence-128 - fixed-128',g('sentence-128'),g('fixed-128')),
('fixed-64 - fixed-32',g('fixed-64'),g('fixed-32')),('fixed-128 - fixed-64',g('fixed-128'),g('fixed-64')),('fixed-256 - fixed-128',g('fixed-256'),g('fixed-128')),('whole - fixed-64',g('whole'),g('fixed-64')),
('bm25 - tfidf (pooled)',pool(st,'bm25'),pool(st,'tfidf')),('bm25 - lsa (pooled)',pool(st,'bm25'),pool(st,'lsa')),('hybrid - bm25 (pooled)',pool(st,'hybrid'),pool(st,'bm25')),
('norm - raw (pooled BM25)',pool('norm','bm25'),pool('raw','bm25')),('norm_stop - norm (pooled BM25)',pool('norm_stop','bm25'),pool('norm','bm25')),('stem(norm_stem) - norm (pooled BM25)',pool('norm_stem','bm25'),pool('norm','bm25')),('norm_stop_stem - norm_stop (pooled BM25)',pool(st,'bm25'),pool('norm_stop','bm25')),
('char4 - stem (pooled BM25)',pool('char4','bm25'),pool(st,'bm25')),('char4 - raw (pooled BM25)',pool('char4','bm25'),pool('raw','bm25')),
('fusion_lex - c4_bm25 (pooled)',pool('mix','fusion_lex'),pool('mix','c4_bm25')),('c4_hybrid - c4_bm25 (pooled)',pool('mix','c4_hybrid'),pool('mix','c4_bm25'))]
rows=[]
for n,a,b in T:
    d,lo,hi=cboot(a,b); rows.append(dict(contrast=n,diff=d,cluster_lo=lo,cluster_hi=hi,excludes_zero=(lo>0)or(hi<0)))
pd.DataFrame(rows).to_csv('results/contrasts_cluster_bootstrap.csv',index=False)
pd.set_option('display.width',250)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(); print(pd.DataFrame(rows[:0]))
V=pd.read_csv('results/variance_decomposition.csv'); print(V[(V.pipeline=='norm_stop_stem')].round(3).drop(columns=['pipeline']).to_string(index=False))
print(pd.read_csv('results/variance_decomposition_restricted.csv').query("pipeline=='norm_stop_stem'").round(3).drop(columns=['pipeline']).to_string(index=False))
V3=pd.read_csv('results/variance_decomposition_3way.csv'); print(V3[V3.pipeline=='norm_stop_stem'].round(3).to_string(index=False))
