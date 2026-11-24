# Dense / hybrid retrieval add-on (run on Colab GPU)

Goal: test whether neural embeddings change the findings (hybrid = lexical + dense), under the same chunking grid and budget metrics.

## Colab cells
```
# 1) upload dense_study.zip, then:
!unzip -q dense_study.zip
%cd dense_study
!ls   # must show run_dense.py, data, corpus_lib.py
!pip -q install sentence-transformers scikit-learn scipy numpy pandas

# 2) quick check (about 2 min): should print B256 bm25_stem = 0.785 for fixed-64 on ARCD
!python run_dense.py --dataset arcd --model intfloat/multilingual-e5-base --configs fixed-64 --out test.pkl.gz

# 3) full runs (GPU: roughly 10-25 min each)
!python run_dense.py --dataset arcd --model intfloat/multilingual-e5-base --out arcd_e5base.pkl.gz
!python run_dense.py --dataset tydi --model intfloat/multilingual-e5-base --out tydi_e5base.pkl.gz
# optional, stronger model:
!python run_dense.py --dataset arcd --model BAAI/bge-m3 --out arcd_bgem3.pkl.gz
!python run_dense.py --dataset tydi --model BAAI/bge-m3 --out tydi_bgem3.pkl.gz

# 4) download the small result files
from google.colab import files
import glob
for f in glob.glob('*_*.pkl.gz'): files.download(f)
```
Upload the *.pkl.gz files in the chat. Each is a few MB (per-question metrics only).

## Sanity targets (BM25 reference systems, must match)
ARCD fixed-64: bm25_stem Budget-256 = 0.785 ; sentence-128 = 0.835 ; whole = 0.337
TyDi fixed-64: bm25_stem Budget-256 = 0.857

## Notes
- e5 models use "query: " / "passage: " prefixes (handled). e5 max length 512 tokens: ARCD whole-document (and some 256-word) chunks are truncated for e5; bge-m3 uses 1024.
- Systems saved: bm25_stem, bm25_c4, dense, rrf_stem_dense, rrf_c4_dense, rrf_c4_dense_w2 (char4 weight 2), rrf_stem_c4_dense.
- If the GPU runs out of memory lower --bs (e.g. 16).
- Do not change anything else: configs, metrics and seeds must stay identical to the main study.

## Analysis of the returned results (already run for the paper)
analysis/analysis_dense.py <arcd|tydi> <pkl.gz> multilingual-e5-base <prefix>  -> contrasts, ranking (Holm), variance decomposition
analysis/ablation.py  -> leave-one-out ablation (Table 11, Fig. 14) and worst-case regret (Table 10, Fig. 13)
analysis/pipeline_fig.py -> Fig. 15
results_dense/ holds the e5-base result files and all CSVs used in Sections 6-7 of the paper.
(analysis scripts expect arcd_rows.json / tydi_rows.json (see data/) and corpus_lib.py/chunkers.py in the working directory.)

## Round 2 (second embedding model + better fusion)  -- run these, in this order
```
%cd /content/dense_study
# new systems added: dense-weighted RRF (_d2, _d4) and z-score fusion (dense weight 0.3 / 0.5 / 0.7 / 0.85)
!python run_dense.py --dataset arcd --model BAAI/bge-m3 --bs 16 --out arcd_bgem3.pkl.gz
!python run_dense.py --dataset tydi --model BAAI/bge-m3 --bs 16 --out tydi_bgem3.pkl.gz
!python run_dense.py --dataset arcd --model intfloat/multilingual-e5-base --out arcd_e5base_v2.pkl.gz
!python run_dense.py --dataset tydi --model intfloat/multilingual-e5-base --out tydi_e5base_v2.pkl.gz
!zip -q results_round2.zip arcd_bgem3.pkl.gz tydi_bgem3.pkl.gz arcd_e5base_v2.pkl.gz tydi_e5base_v2.pkl.gz
from google.colab import files
files.download('results_round2.zip')
```
bge-m3 is about 2 GB to download and takes longer (expect 20-40 min per dataset on a T4); if a run is interrupted, re-run just that line.

## Round 2 results (already analysed in the paper, Section 6.5)
results_round2/: per-question results for e5-base and bge-m3 (13 systems incl. dense-weighted RRF and z-score fusion),
round2.py (pooled results + regret), round2_stats.py (cluster-bootstrap contrasts + split-sample selection check), fig16.py (Figure 16).

## Final ablation (paper Section 8)
results_round2/ablation2.py -> leave-one-out ablation of the proposed pipeline (sentence-64, char-4-gram BM25 + bge-m3, z-score fusion 0.85); ablation2.csv; pipeline_fig2.py -> Figure 15.
The earlier ablation (equal-weight RRF + e5) is kept in results_dense/ablation.csv for reference.
paper_figures/: fig_pipeline_final.png (Fig. 15), fig_ablation_final.png (Fig. 16), fig_fusion.png (Fig. 14); arcd_fig*.png / tydi_fig*.png (Figs. 1-12); fig_dense_vs_lexical.png (Fig. 13).

## M3 + budget curves (reviewer follow-up)  -- run_m3.py
Tests whether the paper's score-fusion result depends on WHERE the z-score is computed (all chunks vs top-K candidates, z-score vs min-max)
and measures Hit@Budget for 10 budgets (32..1024 words) for all systems, on 9 chunking configurations.
Colab cells (GPU runtime):
```
%cd /content/dense_study
!python run_m3.py --dataset arcd --model intfloat/multilingual-e5-base --out m3_arcd_e5.pkl.gz
!python run_m3.py --dataset tydi --model intfloat/multilingual-e5-base --out m3_tydi_e5.pkl.gz
!python run_m3.py --dataset arcd --model BAAI/bge-m3 --bs 16 --out m3_arcd_bgem3.pkl.gz
!python run_m3.py --dataset tydi --model BAAI/bge-m3 --bs 16 --out m3_tydi_bgem3.pkl.gz
!zip -q results_m3.zip m3_*.pkl.gz
from google.colab import files
files.download('results_m3.zip')
```
Sanity targets (Hit@256w; old systems must reproduce):
ARCD e5  fixed-64: bm25_c4 0.815 dense 0.802 rrf_equal 0.839 zall_a85 0.852 | sentence-64: 0.860 0.878 0.898 0.909
ARCD bge fixed-64: 0.815 0.824 0.850 0.865 | sentence-64: 0.860 0.886 0.906 0.909
TyDi e5  fixed-64: 0.884 0.956 0.926 0.954 | sentence-64: 0.885 0.954 0.927 0.957
TyDi bge fixed-64: 0.884 0.948 0.935 0.955 | sentence-64: 0.885 0.951 0.935 0.959
