# Chunking, Retrieval and Fusion for Arabic RAG

Code and results for the paper **"Chunking, Retrieval and Fusion for Arabic RAG: A Budget-Controlled Empirical Study on ARCD and TyDi QA"**.

The study compares 22 chunking configurations (fixed, sliding window with 25%/50% overlap, sentence-aware, paragraph, whole document; 16–256 words) across lexical (BM25, TF-IDF, LSA, char-4-gram BM25), dense (multilingual-e5-base, BAAI/bge-m3) and hybrid (RRF, z-score fusion) retrievers and several Arabic preprocessing pipelines. Systems are compared with a **context-budget metric (Hit@Budget)**: fill a budget of B words with the top-ranked chunks and count a hit if any of them contains the answer span. Findings are replicated on ARCD and TyDi QA (Arabic GoldP).

Proposed pipeline: sentence-aware 64-word chunks + char-4-gram BM25 + bge-m3 + z-score fusion (dense weight 0.85) — Hit@256w = 0.909 (ARCD) / 0.959 (TyDi QA).

## Repository layout

| Folder | Contents |
|---|---|
| `common/` | Shared library: Arabic preprocessing (`arnlp.py`), chunkers (`chunkers.py`), BM25/TF-IDF/LSA/RRF + evaluation (`retrieval.py`), dataset builders (`corpus_lib.py`), pure-Python Parquet reader (`pq.py`, `thrift_compact.py`), paths (`paths.py`) |
| `data/` | `arcd_rows.json`, `tydi_rows.json` – flat question/context/answer-span tables derived from the public datasets (see Data & licences) |
| `stage1_arcd_lexical/` | Lexical study on ARCD: 22 chunk configs × BM25/TF-IDF/LSA/RRF, preprocessing ablation, sensitivity, statistics, figures 1–6 |
| `stage2_tydi_replication/` | Same study on TyDi QA Arabic (incl. offset repair), extra replication analyses |
| `stage3_dense_fusion/` | Dense encoders + fusion (GPU, run on Colab), analysis, ablation, proposed-pipeline figure, round-2 (bge-m3) stats |
| `stage4_review_followup/` | Reviewer-driven follow-ups: M7 lexical-overlap analysis, M2 budget curves, M3 fusion-normalisation study |

Result files (`*.csv`, `*.png`, and `*.pkl.gz` with per-question metrics for every system) are included so every table/figure can be regenerated **without a GPU**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python ≥ 3.10. The GPU step (dense embeddings) additionally needs `torch`, `transformers`, `sentence-transformers` (see `requirements.txt`) and is meant for a Colab T4.

## Reproducing the results

Every script can be started from any directory (it changes into its own folder).

### Stage 1 – ARCD lexical study (CPU)
```bash
cd stage1_arcd_lexical
python make_rows.py <dir with ARCD train/validation parquet>   # optional: data/arcd_rows.json is already provided
python run_experiments.py norm_stop,norm_stop_stem             # main grid
python run_experiments.py raw,norm,norm_stem,char4             # preprocessing ablation
python merge_results.py res_norm_stop,norm_stop_stem.pkl res_raw,norm,norm_stem,char4.pkl   # -> res_all.pkl
python run_exp3.py                                             # char-4-gram + lexical fusion -> res_exp3.pkl
python run_sens.py                                             # sensitivity (BM25 k1/b, LSA dim, RRF k/weights)
python analysis.py && python figures.py                        # CSVs + figures in results/
```

### Stage 2 – TyDi QA replication (CPU)
```bash
cd stage2_tydi_replication
python fix_offsets.py raw_tydi.json ../data/tydi_rows.json     # only if you start from the raw 1,500-question dump
python run_tydi.py            # -> res_all.pkl, res_exp3.pkl
python analysis.py && python rep_extra.py && python run_sens_tydi.py && python figures.py
```

### Stage 3 – dense retrieval and fusion
GPU part (Colab, runtime = T4):
```python
!git clone https://github.com/<IbrahimEldesoky>/<arabic-rag-chunking>.git
%cd <YOUR-REPO>/stage3_dense_fusion
!pip -q install sentence-transformers
!python run_dense.py --dataset arcd --model intfloat/multilingual-e5-base --out arcd_e5base.pkl.gz
!python run_dense.py --dataset tydi --model intfloat/multilingual-e5-base --out tydi_e5base.pkl.gz
!python run_dense.py --dataset arcd --model BAAI/bge-m3 --bs 16 --out arcd_bgem3.pkl.gz
!python run_dense.py --dataset tydi --model BAAI/bge-m3 --bs 16 --out tydi_bgem3.pkl.gz
!python run_m3.py     --dataset arcd --model BAAI/bge-m3 --bs 16 --out m3_arcd_bgem3.pkl.gz   # fusion-normalisation study
```
Add `--mock` to any command to smoke-test without downloading a model. See `stage3_dense_fusion/COLAB_NOTES.md` for sanity targets.

CPU analysis (uses the shipped `*.pkl.gz`):
```bash
cd stage3_dense_fusion
python analysis_dense.py arcd arcd_e5base.pkl.gz multilingual-e5-base arcd_e5   # args: dataset pkl encoder-tag out-prefix
python ablation.py && python round2_stats.py && python ablation2.py
python pipeline_fig.py && python pipeline_fig2.py && python fig16.py
```

### Stage 4 – reviewer follow-ups (CPU)
```bash
cd stage4_review_followup
python m7_overlap.py     # lexical overlap vs dense advantage
python m2_budget.py && python m2_dense.py   # Hit@Budget curves, 32–1024 words
python m3_analysis.py    # fusion normalisation (z-all / z-top-K / min-max) and regret
```

## Data & licences
* **ARCD** (Arabic Reading Comprehension Dataset, Mozannar et al., 2019) – MIT licence. Source: Hugging Face `hsseinmz/arcd`.
* **TyDi QA, GoldP Arabic** (Clark et al., 2020) – Apache-2.0. 657 of the 1,500 answer offsets were repaired by relocating the answer string to the nearest occurrence of the annotated start (`stage2_tydi_replication/fix_offsets.py`).
* `data/*.json` are derived, re-formatted copies; credit the original datasets when you use them.
* Encoders: `intfloat/multilingual-e5-base` (MIT), `BAAI/bge-m3` (MIT).

## Metrics
MRR@10, nDCG@10, document-level hit, answerability ceiling, and **Hit@Budget** (B ∈ {32…1024} words). Statistics: paired bootstrap, cluster bootstrap by article/passage, Holm correction, variance decomposition (chunking vs retriever), regret/minimax, split-sample selection check.

## Citation
If you use this code, please cite the paper (add the final reference here).

## Code licence
MIT (see `LICENSE`) – replace the name in `LICENSE` with yours.
