import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(14,6.2)); ax.set_xlim(0,142); ax.set_ylim(-10,57); ax.axis('off')
def box(x,y,w,h,t,fc,ec='#444',fs=9,bold=False,ls='-'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fc,ec=ec,lw=1.3,ls=ls))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.25)
def arr(x1,y1,x2,y2,c='#333',ls='-'): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.5,color=c,ls=ls))
def note(x,y,t,c='#555',ha='center'): ax.text(x,y,t,ha=ha,va='top',fontsize=7.6,color=c,style='italic',linespacing=1.3)
# main row
box(1,30,15,11,'Arabic\ncorpus','#eef3fb'); box(21,30,19,11,'Normalize\ndiacritics, alef / ya /\nta marbuta, digits','#eef3fb',fs=8.4)
box(45,30,20,11,'Sentence-aware\nchunker\n(about 64 words)','#fff4dc',bold=True,fs=8.8)
box(72,41,24,9.5,'Lexical index\nchar 4-gram BM25','#e5f4e4',bold=True,fs=8.8); box(72,20.5,24,9.5,'Dense index\nmultilingual-e5-base','#fbebeb',bold=True,fs=8.8)
box(101,30,18,11,'RRF fusion\nk = 60\nequal weights','#efedff',bold=True,fs=8.8)
box(101,11,18,11,'Budget packer\ntop chunks until\nN words are used','#f0efec',fs=8.6)
box(101,-8,18,9,'LLM generator\n(not evaluated)','#ffffff',ec='#999',ls='--',fs=8.4)
# question
box(45,8,20,9,'Question','#eef3fb')
arr(16,35.5,21,35.5); arr(40,35.5,45,35.5); arr(65,37.5,72,45.8); arr(65,33.5,72,25.2)
arr(96,45.8,101,38.5); arr(96,25.2,101,32.5); arr(110,30,110,22); arr(110,11,110,1.2)
arr(65,15,72,22,c='#c0392b'); arr(65,13.5,72,43,c='#c0392b')
# annotations (ablation evidence)
note(55,28,'64 words: going to 32 costs\n-6.3 (TyDi) / -7.1 (ARCD)\nsentence boundaries: -5.9 ARCD,\nabout 0 on TyDi')
note(84,18.5,'removing it: -4.2 TyDi, -3.7 ARCD')
note(84,39.8,'removing it: -2.0 ARCD, +2.7 TyDi\nchar 4-grams beat stems by 1.3 / 1.2')
note(121,38,'smallest worst-case\nregret across the two\ncollections (2.2 points)',ha='left')
note(121,19,'evaluate with Hit@Budget,\nnot unconstrained MRR',ha='left')
ax.text(70,55.5,'Recommended Arabic retrieval pipeline (derived from the ablation; changes in Hit@256w)',ha='center',fontsize=10.5,fontweight='bold')
ax.text(3,2,'Red arrows: query path (same normalization as the corpus).\nNumbers: change in Hit@256w when the component is removed\nor replaced (ARCD / TyDi QA, leave-one-out ablation).',fontsize=7.8,color='#555',va='top')
plt.savefig('fig_pipeline.png',dpi=150,bbox_inches='tight',facecolor='white'); print('ok')
