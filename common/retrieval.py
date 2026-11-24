import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2norm
from arnlp import tokens

class Index:
    """Builds all retrievers for one (chunk set, preprocessing mode)."""
    def __init__(s,chunk_texts,mode,k1=1.2,b=0.75,lsa_dim=128,seed=0):
        s.mode=mode; s.k1=k1; s.b=b
        s.cv=CountVectorizer(analyzer=lambda t: tokens(t,mode)); s.tf=s.cv.fit_transform(chunk_texts).tocsr().astype(np.float64)
        N=s.tf.shape[0]; df=np.asarray((s.tf>0).sum(0)).ravel()
        s.idf=np.log(1+(N-df+0.5)/(df+0.5))
        dl=np.asarray(s.tf.sum(1)).ravel(); avg=dl.mean()
        W=s.tf.copy(); rows=np.repeat(np.arange(N),np.diff(W.indptr))
        tfv=W.data; W.data=s.idf[W.indices]*tfv*(k1+1)/(tfv+k1*(1-b+b*dl[rows]/avg))
        s.W=W.T.tocsr()                                   # terms x docs
        s.tfidf=TfidfTransformer(sublinear_tf=True); s.X=s.tfidf.fit_transform(s.tf)   # l2-normalised
        s.svd=None
        if lsa_dim>0:
            nc=min(lsa_dim,N-1,s.X.shape[1]-1)
            s.svd=TruncatedSVD(n_components=nc,random_state=seed,n_iter=7); s.Z=l2norm(s.svd.fit_transform(s.X))
    def scores(s,queries):
        Q=s.cv.transform(queries).astype(np.float64); Qb=Q.copy(); Qb.data[:]=1.0   # binary query terms
        bm25=(Qb@s.W).toarray()
        Xq=s.tfidf.transform(Q); tfidf=(Xq@s.X.T).toarray()
        if s.svd is None: return dict(bm25=bm25,tfidf=tfidf)
        Zq=l2norm(s.svd.transform(Xq)); lsa=Zq@s.Z.T
        return dict(bm25=bm25,tfidf=tfidf,lsa=lsa)

def rank_matrix(S,seed=0):
    rng=np.random.default_rng(seed); noise=rng.random(S.shape[1])*1e-9
    order=np.argsort(-(S+noise),axis=1,kind='stable')
    return order
def rrf(S_list,k=60,seed=0,weights=None):
    fused=np.zeros_like(S_list[0]); weights=weights or [1.0]*len(S_list)
    for S,w in zip(S_list,weights):
        order=rank_matrix(S,seed); ranks=np.empty_like(order); n=S.shape[1]
        np.put_along_axis(ranks,order,np.arange(1,n+1)[None,:].repeat(S.shape[0],0),axis=1)
        fused+=w/(k+ranks)
    return fused

def evaluate(S,chunks,queries,budgets=(128,256,512),ks=(1,3,5,10),seed=0):
    """S: Q x D score matrix. chunks: list (doc,start,end). queries: list of dict. Returns dict of per-query arrays."""
    cd=np.array([c[0] for c in chunks]); cs=np.array([c[1] for c in chunks]); ce=np.array([c[2] for c in chunks])
    cw=np.array([0]*len(chunks))
    order=rank_matrix(S,seed)[:,:200]; Q=len(queries); out={f'hit@{k}':np.zeros(Q) for k in ks}
    for n in ('mrr@10','ndcg@10','doc_hit@1','doc_hit@5','doc_mrr@10')+tuple(f'budget{B}' for B in budgets): out[n]=np.zeros(Q)
    wlen=evaluate.wlen
    disc=1/np.log2(np.arange(2,12))
    for i,q in enumerate(queries):
        rel_all=(cd==q['doc'])&(cs<=q['a_s'])&(ce>=q['a_e'])
        o=order[i]; r=rel_all[o[:10]]; dd=(cd[o[:10]]==q['doc'])
        for k in ks: out[f'hit@{k}'][i]=r[:k].any()
        if r.any(): out['mrr@10'][i]=1/(1+np.argmax(r))
        nrel=min(rel_all.sum(),10)
        if nrel>0: out['ndcg@10'][i]=(r*disc[:len(r)]).sum()/disc[:nrel].sum()
        out['doc_hit@1'][i]=dd[:1].any(); out['doc_hit@5'][i]=dd[:5].any()
        if dd.any(): out['doc_mrr@10'][i]=1/(1+np.argmax(dd))
        cum=np.cumsum(evaluate.wlen[o[:60]])
        for B in budgets:
            m=int(np.searchsorted(cum,B,side='right'))        # number of chunks fitting entirely within budget
            out[f'budget{B}'][i]=rel_all[o[:m]].any() if m>0 else 0
    return out
evaluate.wlen=None
