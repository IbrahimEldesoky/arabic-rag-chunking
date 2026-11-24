import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import gzip,pickle,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
exec(open('round2.py').read().split("# reproducibility")[0])
rng=np.random.default_rng(21)
SYS=['bm25_stem','bm25_c4','dense','rrf_c4_dense','rrf_c4_dense_w2','rrf_c4_dense_d2','rrf_c4_dense_d4','z_c4_dense_a30','z_c4_dense_a50','z_c4_dense_a70','z_c4_dense_a85','rrf_stem_dense','rrf_stem_c4_dense']
def info(ds):
    qs=Qs[ds]; docid=np.array([q['doc'] for q in qs]); Dn=docid.max()+1; mem=[np.where(docid==x)[0] for x in range(Dn)]; return qs,docid,Dn,mem
def pooled_q(ds,m,s,key='budget256'):
    R,t=D[(ds,m)]; cf=[c for c in sorted({k[0] for k in R}) if c!='whole']; return np.mean([R[(c,t,s)][key].astype(float) for c in cf],0)
def cboot(a,b,ds,B=3000):
    qs,docid,Dn,mem=info(ds); dd=a-b; sums=np.array([dd[x].sum() for x in mem]); cnt=np.array([len(x) for x in mem])
    o=np.array([(lambda i:sums[i].sum()/cnt[i].sum())(rng.integers(0,Dn,Dn)) for _ in range(B)]); return dd.mean()*100,np.percentile(o,2.5)*100,np.percentile(o,97.5)*100
rows=[]
for ds in ['ARCD','TyDi']:
    for m in ['e5','bge-m3']:
        a85,a70=pooled_q(ds,m,'z_c4_dense_a85'),pooled_q(ds,m,'z_c4_dense_a70'); dn,rr,bc=pooled_q(ds,m,'dense'),pooled_q(ds,m,'rrf_c4_dense'),pooled_q(ds,m,'bm25_c4')
        for name,x,y in [('z a0.70 - RRF equal',a70,rr),('z a0.85 - RRF equal',a85,rr),('z a0.70 - dense',a70,dn),('z a0.85 - dense',a85,dn),('z a0.70 - char4 BM25',a70,bc),('z a0.85 - char4 BM25',a85,bc)]:
            d,lo,hi=cboot(x,y,ds); rows.append(dict(dataset=ds,model=m,contrast=name,diff=d,lo=lo,hi=hi))
    d,lo,hi=cboot(pooled_q(ds,'bge-m3','dense'),pooled_q(ds,'e5','dense'),ds); rows.append(dict(dataset=ds,model='bge-m3 vs e5',contrast='dense: bge-m3 - e5',diff=d,lo=lo,hi=hi))
    d,lo,hi=cboot(pooled_q(ds,'bge-m3','z_c4_dense_a85'),pooled_q(ds,'e5','z_c4_dense_a85'),ds); rows.append(dict(dataset=ds,model='bge-m3 vs e5',contrast='z a0.85: bge-m3 - e5',diff=d,lo=lo,hi=hi))
C=pd.DataFrame(rows); C.to_csv('round2_contrasts.csv',index=False); print(C.round(1).to_string(index=False))
# split-sample selection (2-fold by document), 300 random splits
print('\nsplit-sample selection (select system on one half of documents, evaluate on the other):')
out=[]
for ds in ['ARCD','TyDi']:
    qs,docid,Dn,mem=info(ds)
    for m in ['e5','bge-m3']:
        M=np.stack([pooled_q(ds,m,s) for s in SYS],1)  # Q x S
        reg_sel=[];reg_fix={s:[] for s in SYS};picked={}
        for _ in range(300):
            perm=rng.permutation(Dn); half=set(perm[:Dn//2]); mask=np.array([d in half for d in docid])
            for tr,te in [(mask,~mask),(~mask,mask)]:
                best=int(M[tr].mean(0).argmax()); picked[SYS[best]]=picked.get(SYS[best],0)+1
                tm=M[te].mean(0); reg_sel.append(tm.max()-tm[best])
                for i,s in enumerate(SYS): reg_fix[s].append(tm.max()-tm[i])
        out.append(dict(dataset=ds,model=m,selected_regret=np.mean(reg_sel)*100,rrf_equal=np.mean(reg_fix['rrf_c4_dense'])*100,dense=np.mean(reg_fix['dense'])*100,z70=np.mean(reg_fix['z_c4_dense_a70'])*100,z85=np.mean(reg_fix['z_c4_dense_a85'])*100,top_pick=max(picked,key=picked.get),pick_share=max(picked.values())/sum(picked.values())))
S=pd.DataFrame(out); S.to_csv('round2_split_sample.csv',index=False); print(S.round(2).to_string(index=False))
