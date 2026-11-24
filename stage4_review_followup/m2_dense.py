import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import gzip,pickle,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
F={('ARCD','e5'):'m3_arcd_e5.pkl.gz',('ARCD','bge-m3'):'m3_arcd_bgem3.pkl.gz',('TyDi','e5'):'m3_tydi_e5.pkl.gz',('TyDi','bge-m3'):'m3_tydi_bgem3.pkl.gz'}
D={k:pickle.load(gzip.open(f,'rb')) for k,f in F.items()}
BUD=(32,64,96,128,192,256,384,512,768,1024); CFG=['fixed-32','fixed-64','fixed-128','fixed-256','sliding-64-ov25','sentence-64','sentence-128','paragraph','whole']
def curve(ds,m,s): R=D[(ds,m)]['R']; t=D[(ds,m)]['tag']; return pd.DataFrame({B:{c:R[(c,t,s)][f'budget{B}'].mean()*100 for c in CFG} for B in BUD})
pd.set_option('display.width',230)
rows=[]
for ds in ['ARCD','TyDi']:
    for m in ['e5','bge-m3']:
        for s in ['dense','ztop100_a70','zall_a85','rrf_equal']:
            c=curve(ds,m,s); 
            for B in BUD: rows.append(dict(dataset=ds,model=m,system=s,budget=B,best=c[B].idxmax(),best_hit=c[B].max(),whole=c.loc['whole',B],spread_wo_32=c.drop(['fixed-32','whole'])[B].max()-c.drop(['fixed-32','whole'])[B].min()))
R=pd.DataFrame(rows); R.to_csv('m2_dense_curves_summary.csv',index=False)
for ds,m,s in [('ARCD','bge-m3','dense'),('ARCD','bge-m3','ztop100_a70'),('TyDi','bge-m3','dense'),('TyDi','bge-m3','ztop100_a70')]:
    print(f'\n{ds} {m} {s}: Hit@Budget % (rows=config, cols=budget in words)'); print(curve(ds,m,s).round(1).to_string())
print('\n== best config per budget (and spread among 7 non-extreme configs: all minus fixed-32 and whole)')
for ds in ['ARCD','TyDi']:
    for m in ['e5','bge-m3']:
        for s in ['dense','ztop100_a70']:
            d=R[(R.dataset==ds)&(R.model==m)&(R.system==s)].set_index('budget')
            print(ds,m,s,'| best:',{B:d.best[B] for B in BUD},'| spread@128/256/512/1024:',[round(d.spread_wo_32[B],1) for B in (128,256,512,1024)])
fig,ax=plt.subplots(2,2,figsize=(11,7),sharey=True)
for j,ds in enumerate(['ARCD','TyDi']):
    for i,(s,lab) in enumerate([('dense','dense (bge-m3)'),('ztop100_a70','score fusion, weight 0.70 (bge-m3)')]):
        c=curve(ds,'bge-m3',s); a=ax[i][j]
        for cf in CFG: a.plot(BUD,c.loc[cf].values,'o-',ms=3,label=cf)
        a.set_xscale('log',base=2); a.set_xticks(BUD); a.set_xticklabels(BUD,rotation=45,fontsize=8); a.set_title(f'{ds}: {lab}',fontsize=9); a.grid(alpha=.3); a.spines[['top','right']].set_visible(False)
        if i==1: a.set_xlabel('context budget (words)')
        if j==0: a.set_ylabel('Hit@Budget (%)')
ax[0][1].legend(fontsize=7,ncol=2,loc='lower right'); plt.tight_layout(); plt.savefig('m2_dense_budget_curves.png',dpi=140); print('fig ok')
