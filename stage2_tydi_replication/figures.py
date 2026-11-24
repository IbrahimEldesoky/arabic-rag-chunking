import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,pickle,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('analysis.py').read().split('# ---------- chunk stats')[0])
ST=pd.read_csv('results/chunk_stats.csv').set_index('config'); VD=pd.read_csv('results/variance_decomposition.csv'); RV=pd.read_csv('results/variance_decomposition_restricted.csv'); FL=pd.read_csv('results/failure_decomposition.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
SZ=[16,32,64,128,256]; col={'fixed':'#1f77b4','sliding-ov25':'#2ca02c','sliding-ov50':'#98df8a','sentence':'#d62728'}
names={'fixed':[f'fixed-{s}' for s in SZ],'sliding-ov25':[f'sliding-{s}-ov25' for s in SZ],'sliding-ov50':[f'sliding-{s}-ov50' for s in SZ],'sentence':[f'sentence-{s}' for s in SZ]}
# F1: size effect, 4 metrics
fig,ax=plt.subplots(1,4,figsize=(15,3.6),sharey=False)
for a,(k,t) in zip(ax,[('mrr@10','MRR@10 (unconstrained)'),('budget128','Hit @ 128-word budget'),('budget256','Hit @ 256-word budget'),('budget512','Hit @ 512-word budget')]):
    for s,cs in names.items(): a.plot([ST.loc[c,'mean_words'] for c in cs],[M(c,'norm_stop_stem','bm25',k).mean() for c in cs],'o-',color=col[s],label=s,ms=4)
    for c,mk in [('paragraph','s'),('whole','*')]: a.plot(ST.loc[c,'mean_words'],M(c,'norm_stop_stem','bm25',k).mean(),mk,color='k',ms=8,label=c)
    a.set_xscale('log'); a.set_xticks([16,32,64,128,256]); a.set_xticklabels([16,32,64,128,256]); a.minorticks_off(); a.set_xlabel('mean chunk length (words)'); a.set_title(t,fontsize=10); a.grid(alpha=.3)
ax[0].set_ylabel('BM25 (norm+stop+stem)'); ax[0].legend(fontsize=7,loc='lower right'); plt.tight_layout(); plt.savefig('results/fig1_chunk_size_effect.png'); plt.close()
# F2: ceiling
fig,a=plt.subplots(figsize=(5.5,3.6))
for s,cs in names.items(): a.plot(SZ,[ST.loc[c,'ceiling'] for c in cs],'o-',color=col[s],label=s)
a.set_xscale('log',base=2); a.set_xticks(SZ); a.set_xticklabels(SZ); a.set_xlabel('nominal chunk size (words)'); a.set_ylabel('answerability ceiling'); a.set_title('Share of questions whose answer lies fully in some chunk',fontsize=9); a.legend(fontsize=8); a.grid(alpha=.3); plt.tight_layout(); plt.savefig('results/fig2_answer_ceiling.png'); plt.close()
# F3: heatmap
sysl=[('norm_stop_stem','bm25','BM25\n(stem)'),('norm_stop_stem','tfidf','TF-IDF\n(stem)'),('norm_stop_stem','lsa','LSA\n(stem)'),('norm_stop_stem','hybrid','RRF\nBM25+LSA'),('char4','bm25','BM25\n(char 4-gram)'),('mix','fusion_lex','RRF stem-BM25\n+char4-BM25')]
order=[c for c in cfgs if c!='whole']
H=np.array([[M(c,m,r,'budget256').mean() for m,r,_ in sysl] for c in order])
fig,a=plt.subplots(figsize=(8,8)); im=a.imshow(H,cmap='viridis',aspect='auto'); a.set_xticks(range(len(sysl))); a.set_xticklabels([s[2] for s in sysl],fontsize=8); a.set_yticks(range(len(order))); a.set_yticklabels(order,fontsize=8)
for i in range(H.shape[0]):
    for j in range(H.shape[1]): a.text(j,i,f'{H[i,j]:.2f}',ha='center',va='center',color='w' if H[i,j]<.62 else 'k',fontsize=7)
a.set_title('Hit rate under a 256-word context budget'); plt.colorbar(im,fraction=.04); plt.tight_layout(); plt.savefig('results/fig3_heatmap_budget256.png'); plt.close()
# F4: variance shares full vs restricted
fig,ax=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for a,(df,t,cn) in zip(ax,[(VD,'All 21 chunk configs (incl. 16-word chunks)','chunking_share'),(RV,'Restricted: chunks >= 64 words','chunking_share')]):
    d=df[(df.pipeline=='norm_stop_stem')&(df.retrievers=='4 retrievers')]; regs=list(d.regime); x=np.arange(len(regs)); rs='retriever_share'
    ints=d['interaction_share'].values if 'interaction_share' in d else d['interaction'].values
    a.bar(x,d[cn],0.6,label='chunking config',color='#1f77b4'); a.bar(x,d[rs],0.6,bottom=d[cn],label='retriever',color='#ff7f0e'); a.bar(x,ints,0.6,bottom=d[cn].values+d[rs].values,label='interaction',color='#bbb')
    a.set_xticks(x); a.set_xticklabels([r.replace(' (unconstrained)','').replace('Hit@','Hit@\n') for r in regs],fontsize=8); a.set_title(t,fontsize=9)
ax[0].set_ylabel('share of between-system variance'); ax[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(-0.1,-0.15),ncol=3); plt.tight_layout(); plt.savefig('results/fig4_variance_shares.png'); plt.close()
# F5: preprocessing with CI
modes=['raw','norm','norm_stop','norm_stem','norm_stop_stem','char4']; lab=['raw','+normalize','+norm+stopwords','+norm+stem','+norm+stop+stem','char 4-grams']
Y=np.array([np.mean([M(c,m,'bm25','budget256') for c in order],0) for m in modes]); mu=Y.mean(1); idx=rng.integers(0,Q,size=(2000,Q)); bs=np.array([Y[:,i].mean(1) for i in idx]); lo,hi=np.percentile(bs,2.5,0),np.percentile(bs,97.5,0)
fig,a=plt.subplots(figsize=(6.5,3.6)); a.bar(range(6),mu,yerr=[mu-lo,hi-mu],color=['#999','#6baed6','#6baed6','#3182bd','#3182bd','#e6550d'],capsize=3); a.set_xticks(range(6)); a.set_xticklabels(lab,rotation=25,ha='right',fontsize=8)
for i,v in enumerate(mu): a.text(i,v+0.012,f'{v:.3f}',ha='center',fontsize=8)
a.set_ylim(0.5,0.85); a.set_ylabel('Hit@256w (mean over 21 configs)'); a.set_title('Arabic preprocessing ablation (BM25)',fontsize=10); plt.tight_layout(); plt.savefig('results/fig5_preprocessing.png'); plt.close()
# F6: failures
d=FL[FL.system=='bm25 char4'].set_index('config'); cs=['fixed-16','fixed-32','fixed-64','fixed-128','fixed-256','sliding-64-ov25','sentence-64','sentence-128','paragraph','whole']; d=d.loc[cs]
fig,a=plt.subplots(figsize=(9,3.8)); b=np.zeros(len(cs))
for k,cl,l in [('success','#2ca02c','top-1 correct'),('right_doc_wrong_chunk','#ffbb78','right article, wrong chunk'),('answer_fragmented','#d62728','answer split across chunks'),('wrong_document','#7f7f7f','wrong article')]:
    a.bar(range(len(cs)),d[k],bottom=b,color=cl,label=l); b+=d[k].values
a.set_xticks(range(len(cs))); a.set_xticklabels(cs,rotation=30,ha='right',fontsize=8); a.set_ylabel('share of questions'); a.set_title('Where does top-1 retrieval fail? (BM25, char 4-grams)',fontsize=10); a.legend(fontsize=7,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.35)); plt.tight_layout(); plt.savefig('results/fig6_failure_modes.png'); plt.close()
print('ok')
