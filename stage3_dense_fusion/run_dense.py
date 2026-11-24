"""Dense / hybrid retrieval experiments for the chunking study.
Usage (Colab GPU):  python run_dense.py --dataset arcd --model intfloat/multilingual-e5-base --out arcd_e5base.pkl.gz
                    python run_dense.py --dataset tydi --model BAAI/bge-m3 --out tydi_bgem3.pkl.gz
New systems: dense-weighted RRF (_d2,_d4) and score-level z-score fusion (_a30.._a85 = dense weight). --mock uses a hashing encoder (no download) to test the pipeline.  --configs fixed-64 sentence-128 runs a subset."""
import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,time,gzip,pickle,argparse,numpy as np
from corpus_lib import build_arcd,build_tydi
from chunkers import *; from retrieval import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['arcd','tydi']); ap.add_argument('--model',default='intfloat/multilingual-e5-base')
ap.add_argument('--out',required=True); ap.add_argument('--mock',action='store_true'); ap.add_argument('--configs',nargs='*'); ap.add_argument('--max_len',type=int,default=0); ap.add_argument('--bs',type=int,default=64)
a=ap.parse_args()
titles,docs,po,qs=(build_arcd('data/arcd_rows.json') if a.dataset=='arcd' else build_tydi('data/tydi_rows.json')); queries=[q['question'] for q in qs]; Qn=len(qs)
print(a.dataset,'docs',len(docs),'questions',Qn,flush=True)
# ---------------- encoder ----------------
if a.mock:
    from sklearn.feature_extraction.text import HashingVectorizer; from arnlp import tokens
    hv=HashingVectorizer(n_features=512,analyzer=lambda t:tokens(t,'norm_stop_stem'),norm='l2')
    def enc(texts,is_query): return hv.transform(texts).toarray().astype(np.float32)
    tag='mock'
else:
    import torch; from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(a.model,device='cuda' if torch.cuda.is_available() else 'cpu'); 
    if torch.cuda.is_available(): m.half()
    low=a.model.lower(); m.max_seq_length=a.max_len or (1024 if 'bge-m3' in low else 512)
    qp,pp=('query: ','passage: ') if 'e5' in low else ('','')
    def enc(texts,is_query):
        return m.encode([(qp if is_query else pp)+t for t in texts],batch_size=a.bs,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=len(texts)>2000).astype(np.float32)
    tag=a.model.split('/')[-1]
Qe=enc(queries,True); cache={}
def zs(S): return (S-S.mean(1,keepdims=True))/(S.std(1,keepdims=True)+1e-9)
def cat(parts): return {k:np.concatenate([p[k] for p in parts]).astype(np.float32) for k in parts[0]}
R={}; t0=time.time(); BS=250
for cfg in make_configs():
    nm=cfg_name(cfg)
    if a.configs and nm not in a.configs: continue
    ch=chunk_corpus(cfg,docs,po); texts=[docs[d][s:e] for d,s,e in ch]; evaluate.wlen=np.array([len(t.split()) for t in texts])
    new=[t for t in dict.fromkeys(texts) if t not in cache]
    if new:
        E=enc(new,False)
        for t,v in zip(new,E): cache[t]=v.astype(np.float16)
    Ce=np.stack([cache[t] for t in texts]).astype(np.float32)
    IA=Index(texts,'norm_stop_stem',lsa_dim=0); IC=Index(texts,'char4',lsa_dim=0); acc={}
    for s in range(0,Qn,BS):
        qb=queries[s:s+BS]; qsb=qs[s:s+BS]; A=IA.scores(qb)['bm25']; C=IC.scores(qb)['bm25']; D=Qe[s:s+BS]@Ce.T
        zC,zD=zs(C),zs(D)
        sets={'bm25_stem':A,'bm25_c4':C,'dense':D,'rrf_stem_dense':rrf([A,D]),'rrf_c4_dense':rrf([C,D]),'rrf_c4_dense_w2':rrf([C,D],weights=[2.0,1.0]),'rrf_stem_c4_dense':rrf([A,C,D]),
              'rrf_c4_dense_d2':rrf([C,D],weights=[1.0,2.0]),'rrf_c4_dense_d4':rrf([C,D],weights=[1.0,4.0]),
              'z_c4_dense_a30':0.3*zD+0.7*zC,'z_c4_dense_a50':0.5*zD+0.5*zC,'z_c4_dense_a70':0.7*zD+0.3*zC,'z_c4_dense_a85':0.85*zD+0.15*zC}
        for k,S in sets.items(): acc.setdefault(k,[]).append(evaluate(S,ch,qsb))
    for k,p in acc.items(): R[(nm,tag,k)]=cat(p)
    print(f"{nm:18s} chunks={len(ch):6d} B256: bm25_stem={R[(nm,tag,'bm25_stem')]['budget256'].mean():.3f} bm25_c4={R[(nm,tag,'bm25_c4')]['budget256'].mean():.3f} dense={R[(nm,tag,'dense')]['budget256'].mean():.3f} rrf_c4_dense={R[(nm,tag,'rrf_c4_dense')]['budget256'].mean():.3f} [{time.time()-t0:.0f}s]",flush=True)
    with gzip.open(a.out,'wb') as f: pickle.dump(dict(R=R,dataset=a.dataset,tag=tag,model=a.model,n_queries=Qn),f)
print('saved',a.out)
