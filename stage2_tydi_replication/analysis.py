import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,pickle,json,itertools,numpy as np,pandas as pd
from corpus import build; from chunkers import *
titles,docs,po,qs=build(); Q=len(qs)
R=pickle.load(open('res_all.pkl','rb')); R.update(pickle.load(open('res_exp3.pkl','rb')))
cfgs=[cfg_name(c) for c in make_configs()]; CF=make_configs()
OUT='results/'; rng=np.random.default_rng(42)
M=lambda c,m,r,k: R[(c,m,r)][k]
# ---------- chunk stats + per-query answerability ----------
stats=[]; ANS={}
for cfg in CF:
    ch=chunk_corpus(cfg,docs,po); by={}
    for d,a,b in ch: by.setdefault(d,[]).append((a,b))
    ans=np.array([any(a<=q['a_s'] and q['a_e']<=b for a,b in by[q['doc']]) for q in qs]); ANS[cfg_name(cfg)]=ans
    w=[len(docs[d][a:b].split()) for d,a,b in ch]
    stats.append(dict(config=cfg_name(cfg),strategy=cfg[0],size=cfg[1],overlap=cfg[2],n_chunks=len(ch),mean_words=np.mean(w),max_words=max(w),ceiling=ans.mean()))
ST=pd.DataFrame(stats); ST.to_csv(OUT+'chunk_stats.csv',index=False)
# ---------- all results means ----------
rows=[dict(config=c,pipeline=m,retriever=r,**{k:float(v.mean()) for k,v in d.items()}) for (c,m,r),d in R.items()]
ALL=pd.DataFrame(rows).merge(ST[['config','n_chunks','mean_words','ceiling']],on='config'); ALL.to_csv(OUT+'all_results.csv',index=False)
# ---------- bootstrap helpers ----------
def boot_idx(B): return rng.integers(0,Q,size=(B,Q))
def paired(a,b,B=5000):
    d=a-b; idx=boot_idx(B); bs=d[idx].mean(1); p=2*min((bs<=0).mean(),(bs>=0).mean()); p=max(p,1/B)
    return dict(diff=float(d.mean()),lo=float(np.percentile(bs,2.5)),hi=float(np.percentile(bs,97.5)),p=float(p))
# ---------- variance decomposition ----------
RET=['bm25','tfidf','lsa','hybrid']; NOWH=[c for c in cfgs if c!='whole']
def decomp(Y,B=1000):
    # Y: (Q,C,R). shares of between-cell variance
    def sh(Mc):
        g=Mc.mean(); mc=Mc.mean(1); mr=Mc.mean(0); C,Rr=Mc.shape
        ssc=Rr*((mc-g)**2).sum(); ssr=C*((mr-g)**2).sum(); sst=((Mc-g)**2).sum(); ssi=sst-ssc-ssr
        return np.array([ssc,ssr,ssi])/sst
    pt=sh(Y.mean(0)); idx=boot_idx(B); bs=np.array([sh(Y[i].mean(0)) for i in idx])
    return pt,np.percentile(bs,2.5,axis=0),np.percentile(bs,97.5,axis=0)
vd=[]
for mode in ['norm_stop','norm_stop_stem']:
    for rets,lab in [(RET,'4 retrievers'),(['bm25','tfidf'],'lexical only')]:
        for key,reg in [('mrr@10','MRR@10 (unconstrained)'),('budget256','Hit@Budget-256w'),('budget512','Hit@Budget-512w')]:
            Y=np.stack([np.stack([M(c,mode,r,key) for r in rets],1) for c in NOWH],1)   # Q,C,R
            pt,lo,hi=decomp(Y)
            vd.append(dict(pipeline=mode,retrievers=lab,regime=reg,chunking_share=pt[0],chunking_lo=lo[0],chunking_hi=hi[0],retriever_share=pt[1],retriever_lo=lo[1],retriever_hi=hi[1],interaction_share=pt[2],
                           chunk_range=float(Y.mean((0,2)).max()-Y.mean((0,2)).min()),retr_range=float(Y.mean((0,1)).max()-Y.mean((0,1)).min())))
VD=pd.DataFrame(vd); VD.to_csv(OUT+'variance_decomposition.csv',index=False)
# size x strategy x retriever decomposition (pipeline B, budget metrics)
STRAT={'fixed':[f'fixed-{s}' for s in (16,32,64,128,256)],'sliding25':[f'sliding-{s}-ov25' for s in (16,32,64,128,256)],'sliding50':[f'sliding-{s}-ov50' for s in (16,32,64,128,256)],'sentence':[f'sentence-{s}' for s in (16,32,64,128,256)]}
def decomp3(mode,key,B=500):
    arr=np.stack([np.stack([np.stack([M(STRAT[s][i],mode,r,key) for r in RET],1) for i in range(5)],1) for s in STRAT],1)   # Q,S,Z,R
    def sh(Mc):
        g=Mc.mean(); S_,Z_,R_=Mc.shape; ms=Mc.mean((1,2)); mz=Mc.mean((0,2)); mr=Mc.mean((0,1))
        ss=Z_*R_*((ms-g)**2).sum(); sz=S_*R_*((mz-g)**2).sum(); sr=S_*Z_*((mr-g)**2).sum(); st=((Mc-g)**2).sum()
        return np.array([sz,ss,sr,st-sz-ss-sr])/st
    pt=sh(arr.mean(0)); idx=boot_idx(B); bs=np.array([sh(arr[i].mean(0)) for i in idx]); return pt,np.percentile(bs,2.5,0),np.percentile(bs,97.5,0)
v3=[]
for mode in ['norm_stop','norm_stop_stem']:
    for key in ['mrr@10','budget256','budget512']:
        pt,lo,hi=decomp3(mode,key)
        for n,i in zip(['chunk size','chunking strategy (fixed/sliding/sentence)','retriever','interactions'],range(4)): v3.append(dict(pipeline=mode,metric=key,factor=n,share=pt[i],lo=lo[i],hi=hi[i]))
V3=pd.DataFrame(v3); V3.to_csv(OUT+'variance_decomposition_3way.csv',index=False)
# ---------- contrasts ----------
con=[]
def add(group,name,a,b,metric):
    r=paired(a,b); con.append(dict(group=group,contrast=name,metric=metric,**r))
for mode in ['norm_stop_stem']:
  for metric in ['budget256','mrr@10']:
    g=lambda c,r='bm25',m=mode: M(c,m,r,metric)
    for s in (16,32,64,128,256):
        add('overlap',f'sliding-{s} ov25 - fixed-{s}',g(f'sliding-{s}-ov25'),g(f'fixed-{s}'),metric)
        add('overlap',f'sliding-{s} ov50 - fixed-{s}',g(f'sliding-{s}-ov50'),g(f'fixed-{s}'),metric)
        add('overlap',f'sliding-{s} ov50 - ov25',g(f'sliding-{s}-ov50'),g(f'sliding-{s}-ov25'),metric)
        add('boundary',f'sentence-{s} - fixed-{s}',g(f'sentence-{s}'),g(f'fixed-{s}'),metric)
    add('boundary','paragraph - fixed-128',g('paragraph'),g('fixed-128'),metric)
    add('boundary','paragraph - sentence-128',g('paragraph'),g('sentence-128'),metric)
    for a,b in [('fixed-32','fixed-16'),('fixed-64','fixed-32'),('fixed-128','fixed-64'),('fixed-256','fixed-128')]: add('size',f'{a} - {b}',g(a),g(b),metric)
    add('size','sliding-64-ov25 - sliding-128-ov25',g('sliding-64-ov25'),g('sliding-128-ov25'),metric)
    # retrievers pooled over 21 configs (per-query mean over configs) and at 3 representative configs
    pooled=lambda r,m=mode: np.mean([M(c,m,r,metric) for c in NOWH],0)
    for a,b in [('bm25','tfidf'),('bm25','lsa'),('hybrid','bm25'),('hybrid','lsa'),('tfidf','lsa')]: add('retriever (pooled 21 cfgs)',f'{a} - {b}',pooled(a),pooled(b),metric)
    for c in ['fixed-128','sentence-128','sliding-64-ov25']:
        for a,b in [('bm25','tfidf'),('hybrid','bm25')]: add(f'retriever @ {c}',f'{a} - {b}',g(c,a),g(c,b),metric)
    # preprocessing (BM25 pooled over configs)
    pp=lambda m: np.mean([M(c,m,'bm25',metric) for c in NOWH],0)
    for a,b in [('norm','raw'),('norm_stop','norm'),('norm_stem','norm'),('norm_stop_stem','norm_stop'),('norm_stop_stem','norm_stem'),('char4','norm_stop_stem'),('char4','raw')]: add('preprocessing (BM25, pooled)',f'{a} - {b}',pp(a),pp(b),metric)
    # lexical fusion & hybrid with char4
    add('fusion','fusion_lex - c4_bm25 (pooled)',np.mean([M(c,'mix','fusion_lex',metric) for c in NOWH],0),np.mean([M(c,'mix','c4_bm25',metric) for c in NOWH],0),metric)
    add('fusion','c4_hybrid - c4_bm25 (pooled)',np.mean([M(c,'mix','c4_hybrid',metric) for c in NOWH],0),np.mean([M(c,'mix','c4_bm25',metric) for c in NOWH],0),metric)
    add('fusion','fusion3 - c4_bm25 (pooled)',np.mean([M(c,'mix','fusion3',metric) for c in NOWH],0),np.mean([M(c,'mix','c4_bm25',metric) for c in NOWH],0),metric)
CON=pd.DataFrame(con); CON.to_csv(OUT+'contrasts.csv',index=False)
# ---------- best system ranking with Holm ----------
sysd={}
for c in NOWH+['whole']:
    sysd[(c,'bm25 stem')]=M(c,'norm_stop_stem','bm25','budget256'); sysd[(c,'bm25 char4')]=M(c,'char4','bm25','budget256'); sysd[(c,'fusion lex')]=M('%s'%c,'mix','fusion_lex','budget256')
keys=list(sysd); means=np.array([sysd[k].mean() for k in keys]); best=keys[int(means.argmax())]
rk=[]
for k in keys:
    if k==best: continue
    r=paired(sysd[best],sysd[k],B=4000); rk.append(dict(system=f'{k[0]} | {k[1]}',mean=float(sysd[k].mean()),diff_vs_best=-r['diff'],lo=-r['hi'],hi=-r['lo'],p=r['p']))
RK=pd.DataFrame(rk).sort_values('mean',ascending=False); n=len(RK); order=np.argsort(RK.p.values); adj=np.empty(n); run=0
for i,j in enumerate(order): run=max(run,(n-i)*RK.p.values[j]); adj[j]=min(1,run)
RK['p_holm']=adj; RK=pd.concat([pd.DataFrame([dict(system=f'{best[0]} | {best[1]}  (BEST)',mean=float(sysd[best].mean()),diff_vs_best=0,lo=0,hi=0,p=np.nan,p_holm=np.nan)]),RK]); RK.to_csv(OUT+'ranking_budget256.csv',index=False)
# ---------- failure decomposition (top-1) ----------
fl=[]
for c in ['fixed-16','fixed-32','fixed-64','fixed-128','fixed-256','sliding-64-ov25','sentence-64','sentence-128','paragraph','whole']:
    for lab,m,r in [('bm25 stem','norm_stop_stem','bm25'),('bm25 char4','char4','bm25')]:
        d=R[(c,m,r)]; h1=d['hit@1'].astype(bool); dh=d['doc_hit@1'].astype(bool); an=ANS[c]
        fl.append(dict(config=c,system=lab,success=h1.mean(),wrong_document=(~dh).mean(),answer_fragmented=(dh&~an).mean(),right_doc_wrong_chunk=(dh&an&~h1).mean()))
FL=pd.DataFrame(fl); FL.to_csv(OUT+'failure_decomposition.csv',index=False)
pickle.dump(dict(VD=VD,V3=V3,CON=CON,RK=RK,FL=FL,ST=ST),open('results/analysis.pkl','wb'))
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print('== chunk stats'); print(ST.round(3).to_string())
print('\n== variance decomposition (2-way: chunking cfg x retriever)'); print(VD.round(3).to_string())
print('\n== 3-way'); print(V3[V3.pipeline=='norm_stop_stem'].round(3).to_string())
