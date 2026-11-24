import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,gzip,pickle,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from corpus_lib import build_arcd,build_tydi
tag='multilingual-e5-base'; rng=np.random.default_rng(5)
DS={'ARCD':('arcd_e5base.pkl.gz',build_arcd,P_ARCD),'TyDi QA':('tydi_e5base.pkl.gz',build_tydi,P_TYDI)}
FULL=('sentence-64','rrf_c4_dense')
ABL=[('Chunking','sentence boundaries -> fixed windows',('fixed-64','rrf_c4_dense')),('Chunking','sentence boundaries -> 25% overlap windows',('sliding-64-ov25','rrf_c4_dense')),
     ('Chunking','size 64 -> 32 words',('sentence-32','rrf_c4_dense')),('Chunking','size 64 -> 128 words',('sentence-128','rrf_c4_dense')),('Chunking','size 64 -> 256 words',('sentence-256','rrf_c4_dense')),('Chunking','no sentence packing: native paragraphs',('paragraph','rrf_c4_dense')),
     ('Retrieval','remove dense (char 4-gram BM25 only)',('sentence-64','bm25_c4')),('Retrieval','remove lexical (dense only)',('sentence-64','dense')),
     ('Retrieval','char 4-grams -> stemming in lexical branch',('sentence-64','rrf_stem_dense')),('Retrieval','add stemmed BM25 as third ranker',('sentence-64','rrf_stem_c4_dense')),('Retrieval','lexical weight x2 in fusion',('sentence-64','rrf_c4_dense_w2'))]
out=[]; REG=[]; LINES={}
for name,(pk,builder,rows) in DS.items():
    titles,docs,po,qs=builder(rows); Q=len(qs); R=pickle.load(gzip.open(pk,'rb'))['R']; M=lambda c,s,k: R[(c,tag,s)][k].astype(np.float64)
    docid=np.array([q['doc'] for q in qs]); D=docid.max()+1; mem=[np.where(docid==x)[0] for x in range(D)]; cnt=np.array([len(m) for m in mem])
    def cb(a,b,B=3000):
        dd=a-b; s=np.array([dd[m].sum() for m in mem]); o=np.array([(lambda i:s[i].sum()/cnt[i].sum())(rng.integers(0,D,D)) for _ in range(B)]); return float(dd.mean()),float(np.percentile(o,2.5)),float(np.percentile(o,97.5))
    for key in ['budget256','mrr@10','budget512']:
        full=M(FULL[0],FULL[1],key)
        out.append(dict(dataset=name,metric=key,group='-',ablation='FULL: sentence-64 + RRF(char4 BM25, e5)',value=float(full.mean()),delta=0.0,lo=0.0,hi=0.0))
        for g,lab,(c,s) in ABL:
            m,lo,hi=cb(M(c,s,key),full); out.append(dict(dataset=name,metric=key,group=g,ablation=lab,value=float(M(c,s,key).mean()),delta=m,lo=lo,hi=hi))
    NOWH=[c for c in sorted({k[0] for k in R}) if c!='whole']
    syss=['bm25_stem','bm25_c4','dense','rrf_stem_dense','rrf_c4_dense','rrf_c4_dense_w2','rrf_stem_c4_dense']
    pooled={s:float(np.mean([M(c,s,'budget256').mean() for c in NOWH])) for s in syss}; best=max(pooled.values())
    for s in syss: REG.append(dict(dataset=name,system=s,pooled_hit256=pooled[s],regret=best-pooled[s]))
    LINES[name]={s:[M(c,s,'budget256').mean() for c in ['fixed-16','fixed-32','fixed-64','fixed-128','fixed-256']] for s in ['bm25_stem','bm25_c4','dense','rrf_c4_dense']}
    LINES[name]['_sent']={s:[M(f'sentence-{z}',s,'budget256').mean() for z in (16,32,64,128,256)] for s in ['bm25_c4','dense','rrf_c4_dense']}
A=pd.DataFrame(out); A.to_csv('ablation.csv',index=False); G=pd.DataFrame(REG); G.to_csv('regret.csv',index=False)
pd.set_option('display.width',220)
print(A[A.metric=='budget256'].pivot_table(index=['group','ablation'],columns='dataset',values=['delta','lo','hi'],sort=False).round(3).to_string())
print(A[(A.metric=='budget256')&(A.group=='-')][['dataset','value']])
print(G.assign(s=G.system).pivot(index='system',columns='dataset',values=['pooled_hit256','regret']).round(3).to_string())
print('max regret per system:',G.groupby('system').regret.max().round(3).to_dict())
# ---- Fig 7: retrievers x chunk size, two datasets
fig,ax=plt.subplots(1,2,figsize=(11,3.8),sharey=False); SZ=[16,32,64,128,256]; st={'bm25_stem':('BM25 (stem)','#1f77b4','--'),'bm25_c4':('BM25 (char 4-gram)','#2ca02c','-'),'dense':('Dense (e5-base)','#d62728','-'),'rrf_c4_dense':('RRF: char4 BM25 + dense','#9467bd','-')}
for a,name in zip(ax,DS):
    for s,(lab,c,ls) in st.items(): a.plot(SZ,LINES[name][s],'o'+ls,color=c,label=lab,ms=4)
    a.set_xscale('log',base=2); a.set_xticks(SZ); a.set_xticklabels(SZ); a.set_title(name+' (fixed windows)',fontsize=10); a.set_xlabel('chunk size (words)'); a.grid(alpha=.3); a.spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Hit @ 256-word budget'); ax[1].legend(fontsize=8,loc='lower right'); plt.tight_layout(); plt.savefig('fig7_dense_vs_lexical.png',dpi=140); plt.close()
# ---- Fig 8: ablation forest
d=A[(A.metric=='budget256')&(A.group!='-')]; labs=[l for _,l,_ in ABL]; y=np.arange(len(labs))[::-1]
fig,a=plt.subplots(figsize=(9,5))
for off,(name,col) in zip([-0.17,0.17],[('ARCD','#1f77b4'),('TyDi QA','#d62728')]):
    s=d[d.dataset==name].set_index('ablation').loc[labs]; a.errorbar(s.delta*100,y+off,xerr=[(s.delta-s.lo)*100,(s.hi-s.delta)*100],fmt='o',color=col,label=name,capsize=2,ms=4)
a.axvline(0,color='k',lw=.8); a.set_yticks(y); a.set_yticklabels(labs,fontsize=8); a.set_xlabel('change in Hit@256w vs. full pipeline (points, 95% CI)'); a.legend(fontsize=8,loc='lower right'); a.grid(axis='x',alpha=.3); a.spines[['top','right']].set_visible(False)
a.set_title('Ablation of the recommended pipeline (sentence-64 + RRF(char4 BM25, e5))',fontsize=10); plt.tight_layout(); plt.savefig('fig8_ablation.png',dpi=140); plt.close()
