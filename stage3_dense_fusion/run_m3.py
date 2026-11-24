"""M3 + dense budget curves.
(1) Fusion normalisation: z-score over ALL chunks (as in the paper) vs z-score / min-max over the TOP-K candidates (union of each ranker's top-K).
(2) Hit@Budget for 10 budgets (32..1024 words) for every system, on 9 representative chunking configurations.
Usage: python run_m3.py --dataset arcd --model BAAI/bge-m3 --bs 16 --out m3_arcd_bgem3.pkl.gz     (add --mock to test without a model)"""
import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,time,gzip,pickle,argparse,numpy as np
from corpus_lib import build_arcd,build_tydi
from chunkers import *; from retrieval import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['arcd','tydi']); ap.add_argument('--model',default='intfloat/multilingual-e5-base')
ap.add_argument('--out',required=True); ap.add_argument('--mock',action='store_true'); ap.add_argument('--configs',nargs='*'); ap.add_argument('--bs',type=int,default=64); ap.add_argument('--max_len',type=int,default=0)
a=ap.parse_args()
titles,docs,po,qs=(build_arcd('data/arcd_rows.json') if a.dataset=='arcd' else build_tydi('data/tydi_rows.json')); queries=[q['question'] for q in qs]; Qn=len(qs)
print(a.dataset,'docs',len(docs),'questions',Qn,flush=True)
if a.mock:
    from sklearn.feature_extraction.text import HashingVectorizer; from arnlp import tokens
    hv=HashingVectorizer(n_features=512,analyzer=lambda t:tokens(t,'norm_stop_stem'),norm='l2')
    def enc(texts,is_query): return hv.transform(texts).toarray().astype(np.float32)
    tag='mock'
else:
    import torch; from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(a.model,device='cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): m.half()
    low=a.model.lower(); m.max_seq_length=a.max_len or (1024 if 'bge-m3' in low else 512)
    qp,pp=('query: ','passage: ') if 'e5' in low else ('','')
    def enc(texts,is_query): return m.encode([(qp if is_query else pp)+t for t in texts],batch_size=a.bs,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=len(texts)>2000).astype(np.float32)
    tag=a.model.split('/')[-1]
Qe=enc(queries,True); cache={}
CFG=[('fixed',32,0.0),('fixed',64,0.0),('fixed',128,0.0),('fixed',256,0.0),('sliding',64,0.25),('sentence',64,0.0),('sentence',128,0.0),('paragraph',0,0.0),('whole',0,0.0)]
BUD=(32,64,96,128,192,256,384,512,768,1024); KS=(20,100); ALPHAS=(0.5,0.7,0.85)
def zs(S): return (S-S.mean(1,keepdims=True))/(S.std(1,keepdims=True)+1e-9)
def cand_mask(C,D,K):
    m=np.zeros(C.shape,bool); r=np.arange(C.shape[0])[:,None]
    for S in (C,D):
        k=min(K,S.shape[1]-1); idx=np.argpartition(-S,k-1,axis=1)[:,:k]; m[r,idx]=True
    return m
def norm_z(S,m):
    n=m.sum(1,keepdims=True); mu=(S*m).sum(1,keepdims=True)/n; var=(((S-mu)**2)*m).sum(1,keepdims=True)/n; return (S-mu)/(np.sqrt(var)+1e-9)
def norm_mm(S,m):
    lo=np.where(m,S,np.inf).min(1,keepdims=True); hi=np.where(m,S,-np.inf).max(1,keepdims=True); return (S-lo)/(hi-lo+1e-9)
def fuse(C,D,alpha,kind,K=None):
    base=alpha*zs(D)+(1-alpha)*zs(C)
    if kind=='zall': return base
    m=cand_mask(C,D,K); f=norm_z if kind=='z' else norm_mm
    F=alpha*f(D,m)+(1-alpha)*f(C,m)
    return np.where(m,F+1e4,base)      # candidates ranked first by the new normalisation; the rest fall back to the all-chunk score
def systems(C,D):
    S={'bm25_c4':C,'dense':D,'rrf_equal':rrf([C,D]),'rrf_d2':rrf([C,D],weights=[1.0,2.0])}
    for al in ALPHAS:
        t=int(round(al*100)); S[f'zall_a{t}']=fuse(C,D,al,'zall')
        for K in KS: S[f'ztop{K}_a{t}']=fuse(C,D,al,'z',K); S[f'mmtop{K}_a{t}']=fuse(C,D,al,'mm',K)
    return S
KEEP=['hit@1','hit@5','mrr@10','doc_hit@1']+[f'budget{B}' for B in BUD]
R={}; t0=time.time(); BS=250
for cfg in CFG:
    nm=cfg_name(cfg)
    if a.configs and nm not in a.configs: continue
    ch=chunk_corpus(cfg,docs,po); texts=[docs[d][s:e] for d,s,e in ch]; evaluate.wlen=np.array([len(t.split()) for t in texts])
    new=[t for t in dict.fromkeys(texts) if t not in cache]
    if new:
        E=enc(new,False)
        for t,v in zip(new,E): cache[t]=v.astype(np.float16)
    Ce=np.stack([cache[t] for t in texts]).astype(np.float32); IC=Index(texts,'char4',lsa_dim=0); acc={}
    for s in range(0,Qn,BS):
        qb=queries[s:s+BS]; qsb=qs[s:s+BS]; C=IC.scores(qb)['bm25']; D=Qe[s:s+BS]@Ce.T
        for k,S in systems(C,D).items():
            r=evaluate(S,ch,qsb,budgets=BUD); acc.setdefault(k,[]).append({x:r[x] for x in KEEP})
    for k,p in acc.items(): R[(nm,tag,k)]={x:np.concatenate([q[x] for q in p]).astype(np.float32) for x in KEEP}
    g=lambda k:R[(nm,tag,k)]['budget256'].mean()
    print(f"{nm:16s} chunks={len(ch):6d} B256: bm25_c4={g('bm25_c4'):.3f} dense={g('dense'):.3f} rrf_equal={g('rrf_equal'):.3f} zall_a85={g('zall_a85'):.3f} ztop100_a85={g('ztop100_a85'):.3f} mmtop100_a85={g('mmtop100_a85'):.3f} [{time.time()-t0:.0f}s]",flush=True)
    with gzip.open(a.out,'wb') as f: pickle.dump(dict(R=R,dataset=a.dataset,tag=tag,model=a.model,n_queries=Qn,budgets=BUD),f)
print('saved',a.out)
