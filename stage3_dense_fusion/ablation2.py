import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import warnings; warnings.filterwarnings('ignore')
exec(open('round2.py').read().split("# reproducibility")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng=np.random.default_rng(77); ENC='bge-m3'; FULLC='sentence-64'; FULLS='z_c4_dense_a85'
def info(ds):
    docid=np.array([q['doc'] for q in Qs[ds]]); Dn=docid.max()+1; return Dn,[np.where(docid==x)[0] for x in range(Dn)]
def get(ds,m,c,s,k='budget256'):
    R,t=D[(ds,m)]; return R[(c,t,s)][k].astype(float)
def cb(a,b,ds,B=3000):
    Dn,mem=info(ds); dd=a-b; sm=np.array([dd[x].sum() for x in mem]); cn=np.array([len(x) for x in mem])
    o=np.array([(lambda i:sm[i].sum()/cn[i].sum())(rng.integers(0,Dn,Dn)) for _ in range(B)]); return dd.mean()*100,np.percentile(o,2.5)*100,np.percentile(o,97.5)*100
ABL=[('Chunking','sentence boundaries -> fixed windows',ENC,'fixed-64',FULLS),('Chunking','sentence boundaries -> 25% overlap windows',ENC,'sliding-64-ov25',FULLS),
 ('Chunking','size 64 -> 32 words',ENC,'sentence-32',FULLS),('Chunking','size 64 -> 128 words',ENC,'sentence-128',FULLS),('Chunking','size 64 -> 256 words',ENC,'sentence-256',FULLS),('Chunking','native paragraphs',ENC,'paragraph',FULLS),
 ('Retrieval','remove dense branch (char-4-gram BM25 only)',ENC,FULLC,'bm25_c4'),('Retrieval','remove lexical branch (dense only)',ENC,FULLC,'dense'),
 ('Fusion','score fusion -> equal-weight RRF',ENC,FULLC,'rrf_c4_dense'),('Fusion','score fusion -> RRF with dense weight 2',ENC,FULLC,'rrf_c4_dense_d2'),
 ('Fusion','dense weight 0.85 -> 0.70',ENC,FULLC,'z_c4_dense_a70'),('Fusion','dense weight 0.85 -> 0.50',ENC,FULLC,'z_c4_dense_a50'),
 ('Encoder','bge-m3 -> multilingual-e5-base','e5',FULLC,FULLS)]
rows=[]
for ds in ['ARCD','TyDi']:
    for key in ['budget256','mrr@10']:
        full=get(ds,ENC,FULLC,FULLS,key); rows.append(dict(dataset=ds,metric=key,group='-',ablation='FULL',value=full.mean()*100,delta=0,lo=0,hi=0))
        for g,lab,m,c,s in ABL:
            x=get(ds,m,c,s,key); d,lo,hi=cb(x,full,ds); rows.append(dict(dataset=ds,metric=key,group=g,ablation=lab,value=x.mean()*100,delta=d,lo=lo,hi=hi))
A=pd.DataFrame(rows); A.to_csv('ablation2.csv',index=False)
a=A[A.metric=='budget256']; print(a[a.group=='-'][['dataset','value']].round(1).to_string(index=False))
for ds in ['ARCD','TyDi']:
    print(ds); print(a[(a.dataset==ds)&(a.group!='-')][['group','ablation','delta','lo','hi']].round(1).to_string(index=False))
labs=[l for _,l,*_ in ABL]; y=np.arange(len(labs))[::-1]; d=a[a.group!='-']
fig,ax=plt.subplots(figsize=(9.5,5.6))
for off,(name,col) in zip([-0.17,0.17],[('ARCD','#1f77b4'),('TyDi','#d62728')]):
    s=d[d.dataset==name].set_index('ablation').loc[labs]; ax.errorbar(s.delta,y+off,xerr=[s.delta-s.lo,s.hi-s.delta],fmt='o',color=col,label='ARCD' if name=='ARCD' else 'TyDi QA',capsize=2,ms=4)
ax.axvline(0,color='k',lw=.8); ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=8); ax.set_xlabel('change in Hit@256w vs. the proposed pipeline (points, 95% CI)'); ax.legend(fontsize=8,loc='lower right'); ax.grid(axis='x',alpha=.3); ax.spines[['top','right']].set_visible(False)
ax.set_title('Ablation of the proposed pipeline (sentence-64, char-4-gram BM25 + bge-m3, score fusion 0.85)',fontsize=10); plt.tight_layout(); plt.savefig('fig_ablation2.png',dpi=140); print('fig ok')
