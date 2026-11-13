# Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation

Official implementation of **AST (Adaptive Silhouette Tree)**, from the RecSys '26 paper

> Guy Hadad, Haggai Roitman, Bracha Shapira. *Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation.* In Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26), pp. 390–399. [https://doi.org/10.1145/3773078.3831756](https://doi.org/10.1145/3773078.3831756)

AST is a top-down divisive clustering method that builds semantic IDs as **paths in a tree**. The branching factor of every node is chosen by maximizing the silhouette coefficient, so dense regions of the item space get finer partitions and homogeneous regions stay compact. AST also requires every branch to carry a minimum share of interaction mass, so that rare prefixes still receive enough supervision. It enforces this with an exact MILP assignment alternated with mass-weighted center updates (the mass-aware constrained refinement of §4.4.3, reported as AST-V2 in the paper).

This repository contains what is needed to reproduce the AST results of the paper's main table (Table 1): the AST tokenizer, the shared T5 generative recommender, and the evaluation protocol.

---

## Contents

1. [Method](#method)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Data](#data)
5. [Quick start](#quick-start)
6. [Reproducing Table 1](#reproducing-table-1)
7. [Hyperparameter search](#hyperparameter-search)
8. [Pipeline](#pipeline)
9. [Evaluation protocol](#evaluation-protocol)
10. [Configuration reference](#configuration-reference)
11. [Citation](#citation)

---

## Method

A semantic ID for item *i* is a sequence `s_i = (s_i^0, ..., s_i^{m-1})` with `s_i^ℓ ∈ {0, ..., B-1}`, where `m` is the code length and `B` (`base`) is the maximum branching factor. Each position has its own token vocabulary `<cℓ_0> ... <cℓ_{B-1}>`, so the meaning of a token is always relative to its prefix.

AST builds the first `L = m - 1` tokens by recursively clustering unit-norm item-text embeddings (Qwen3-Embedding-0.6B over title, categories and description). The last token enumerates the items that share a prefix, which makes every code unique.

| Component | What it does | Paper |
|---|---|---|
| Adaptive branching | At every node, try `K = K_min, K_min+1, ...` and keep the split with the highest silhouette (Euclidean, early stop after 5 non-improving candidates). `K_min` is capacity-aware, so every final bucket fits into `B` suffix tokens. | §4.2 |
| Minimum-mass constraint `ρ` | Every child must hold at least a fraction `ρ` of its parent's mass, which implies `K ≤ ⌊1/ρ⌋`. Each candidate split is made feasible with an exact MILP assignment (Eq. 1–4); splits that cannot satisfy it are rejected. | §4.4.1 |
| Mass-aware constrained refinement | Starts from mass-weighted KMeans centers, alternates the exact MILP assignment with mass-weighted center updates (a constrained Lloyd iteration), and scores splits with the mass-weighted silhouette. | §4.4.3 |
| Constrained decoding | Beam search is restricted to a trie of the real item codes, so every generated ID is an item in the catalog. | §4.3 |

The MILP is solved with **CBC** through `python-mip`, so no license is needed. **Gurobi** is optional (`tokenization.milp_solver_name=GRB`).

---

## Repository layout

```
AST/
├── train.py             # entry point: data -> AST semantic IDs -> T5 recommender -> test metrics
├── silhouette_tree.py   # AST tokenizer (adaptive branching, minimum-mass MILP, constrained refinement)
├── data.py              # Amazon Reviews 2014 (5-core): download, leave-last-out splits, item text
├── model.py             # T5 encoder-decoder, code trie, trie-constrained Seq2SeqTrainer
├── evaluation.py        # Recall@k / NDCG@k over decoded items
├── utils.py             # seeding, item ids and mass, <cℓ_k> vocabulary, seq2seq examples
├── summarize_runs.py    # rank the runs of a hyperparameter search by validation NDCG@10
├── conf/                # Hydra configs
│   ├── config.yaml
│   ├── data/amazon2014.yaml
│   ├── model/t5.yaml
│   ├── training/default.yaml
│   ├── tokenization/ast.yaml
│   └── experiment/      # Table 1 settings per dataset: beauty, toys, sports, health, home
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Installation

```bash
conda create -n ast python=3.11
conda activate ast
pip install -r requirements.txt        # torch 2.10 (CUDA 12.x build), transformers 5.3, ...
```

The paper's experiments ran on a single NVIDIA RTX 6000 Ada GPU. Training uses bfloat16, so the GPU needs Ampere or newer; on older cards, pass `training.hf.bf16=false`.

---

## Data

We use five domains of **Amazon Reviews 2014** (McAuley et al.), in the standard 5-core release from SNAP. The review and metadata files are downloaded automatically into `raw_data/amazon/` on first use. Histories are truncated to the last **20** interactions, and the item text is the concatenation of **title**, **categories** and **description**.

| Dataset | `data.category` | #Users | #Items | Sparsity | Avg. length |
|---|---|---:|---:|---:|---:|
| Beauty | `Beauty` | 22,363 | 12,101 | 99.93% | 8.88 |
| Toys | `Toys_and_Games` | 19,412 | 11,924 | 99.93% | 8.63 |
| Sports | `Sports_and_Outdoors` | 35,598 | 18,357 | 99.95% | 8.32 |
| Health | `Health_and_Personal_Care` | 38,609 | 18,534 | 99.95% | 8.97 |
| Home | `Home_and_Kitchen` | 66,519 | 28,237 | 99.97% | 8.29 |

---

## Quick start

```bash
# AST on Beauty with the Table 1 hyperparameters
python train.py experiment=beauty

# print the fully resolved config without running
python train.py experiment=beauty --cfg job
```

Each run writes to `outputs/<category>/AST/<timestamp>/`:
- `item2tokseq_ids.pkl`: the semantic ID (a list of `m` token ids) of every item;
- `sids_stats.json`: tree statistics and build time;
- `test_metrics.json`: test Recall@{5,10} and NDCG@{5,10} of the best checkpoint (selected by validation NDCG@10), plus that checkpoint's validation metrics (`val_*`).

---

## Reproducing Table 1

All runs share the same recommender and training budget, so a Table 1 cell is fully defined by the dataset, the code length `m`, the mass constraint `ρ`, and the item mass (training interaction frequency, or uniform). `base = 4096` for every dataset. The per-dataset values are in `conf/experiment/`:

| Dataset | `m` | `ρ` | Item mass | Config | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---|---|---:|---:|---:|---:|
| Beauty | 4 | 0.02 | uniform | `beauty` | 5.12 | 7.51 | 3.51 | 4.28 |
| Toys | ? | ? | ? | `toys` | 5.27 | 7.94 | 3.45 | 4.31 |
| Sports | 5 | 0.02 | interactions | `sports` | 3.09 | 4.62 | 2.03 | 2.51 |
| Health | 5 | 0.05 | interactions | `health` | 4.48 | 6.64 | 2.98 | 3.68 |
| Home | 5 | 0.01 | uniform | `home` | 2.06 | 3.02 | 1.36 | 1.67 |

Metrics are percentages on the test split, as reported in the paper. The Toys configuration for this code release will be released soon.

**Results of this code.** One run per dataset with the settings above (seed 42, one RTX 4090 GPU):

| Dataset | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Beauty | 5.13 | 7.53 | 3.43 | 4.20 |
| Toys | ? | ? | ? | ? |
| Sports | 3.13 | 4.58 | 2.10 | 2.57 |
| Health | 4.47 | 6.62 | 2.95 | 3.65 |
| Home | 1.99 | 2.96 | 1.34 | 1.65 |

On these four datasets, every metric is within 0.08 points of the paper. Toys results will be released soon.

> **Note.** Results can vary from run to run. They depend on the random seed (KMeans initialization, model initialization and data order), and parts of the pipeline are not deterministic: the MILP solves are time-limited, so the semantic IDs can change with machine speed, and GPU kernels differ across GPU models and library versions. Expect small differences from the numbers above.

```bash
# one dataset
python train.py experiment=sports

# all five datasets (sequentially; see below for cost). Toys still uses provisional hyperparameters.
for ds in beauty toys sports health home; do
  python train.py experiment=$ds
done
```

**Cost.** Building the semantic IDs is CPU work dominated by the MILP solves: on 8 cores, 0.5–1.5 h for Beauty and Toys and 1.5–4 h for Sports, Health and Home. Training on one RTX 4090 takes 2–3 h for Beauty and Toys, about 6 h for Sports and Health, and 10–15 h for Home. With the same semantic IDs, GPU model and library versions, training reproduces the same numbers.

**Running the stages separately** (for example, semantic IDs on a CPU node and training on a GPU node):

```bash
python train.py experiment=home stop_after=embed     # GPU: encode + cache item embeddings
python train.py experiment=home stop_after=sids hydra.run.dir=runs/home   # CPU: build semantic IDs
python train.py experiment=home tokenization.sids_path=runs/home/item2tokseq_ids.pkl   # GPU: train + test
```

Item embeddings are cached under `raw_data/embeddings/` (keyed by encoder and item texts), so all configurations of one dataset encode the catalog only once.

---

## Hyperparameter search

Per dataset, AST has three hyperparameters: the code length `m`, the mass constraint `ρ` (`min_share_prob`), and the item mass (`use_item_mass=true` for the training interaction frequency, `false` for uniform). Hydra's multirun runs a grid, and `summarize_runs.py` ranks the runs:

```bash
python train.py -m experiment=toys \
  tokenization.m=4,5 \
  tokenization.min_share_prob=0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1 \
  tokenization.use_item_mass=true,false

python summarize_runs.py multirun/ --csv toys_search.csv
```

Every run records the validation metrics of its selected checkpoint (`val_*`) next to the test metrics, and `summarize_runs.py` sorts the runs by validation NDCG@10. Choose the configuration by validation and report its test metrics.

The grid above has 36 configurations. To run it in parallel, build every configuration's semantic IDs on CPU first (`stop_after=sids`), then train each one on a GPU with `tokenization.sids_path=...` and the same `tokenization.*` overrides, which are recorded in `test_metrics.json`. Configurations whose semantic IDs are identical (for example, when the mass constraint is never active) give identical results, so train those only once.

---

## Pipeline

[train.py](train.py) runs one experiment in a single Hydra `main()`:

1. **Data** ([data.py](data.py)): parse the 5-core reviews, sort every user's interactions by time, build leave-last-out splits, and map ASINs to item ids `3..N+2` (0–2 are `<pad>`, `<eos>`, `<unk>`).
2. **Item text and mass**: `title | categories | description` per item, and the item mass `p_i` = training interaction frequency ([utils.py](utils.py)).
3. **Semantic IDs** ([silhouette_tree.py](silhouette_tree.py)): encode the items, build the AST, and assign the collision-free suffix.
4. **Seq2seq examples**: the encoder input is the concatenated semantic IDs of the history, and the decoder target is the semantic ID of the next item.
5. **Model** ([model.py](model.py)): T5 with 6 encoder and 6 decoder layers (`d_model=128`, `d_ff=1024`, 6 heads), trained from scratch.
6. **Constrained decoding** ([model.py](model.py)): a trie over all item codes restricts every beam-search step to valid continuations, and generation is exactly `m` tokens long.
7. **Evaluation** ([evaluation.py](evaluation.py)): decoded codes are mapped back to items, and Recall@k / NDCG@k are computed.

---

## Evaluation protocol

- **Split**: leave-last-out. The last interaction of each user is the test target, the second-to-last is the validation target, and every earlier prefix is a training example.
- **Metrics**: Recall@k and NDCG@k for `k ∈ {5, 10}`, at the item level. The 10 returned beams (beam size 10) are mapped to items and de-duplicated in rank order. The codebook is checked to be collision-free.
- **Model selection**: evaluation every 10 epochs, the checkpoint with the best validation NDCG@10 is kept, and early stopping uses a patience of 5 evaluations (up to 500 epochs). The reported numbers are on the test split.

---

## Configuration reference

### Tokenization (`conf/tokenization/ast.yaml`)

| Key | Symbol | Value | Meaning |
|---|---|---|---|
| `base` | `B` | 4096 | Maximum branching factor / tokens per level |
| `m` | `m` | per dataset | Semantic ID length (`m-1` tree levels + 1 suffix token) |
| `encoder_name` | | `Qwen/Qwen3-Embedding-0.6B` | Item-text encoder |
| `max_iter_kmeans`, `n_init_kmeans` | | 50, 10 | KMeans budget at every candidate split |
| `patience` | | 5 | Stop the `K` search after this many non-improving candidates |
| `silhouette_sample_size` | | 2000 | Silhouette is estimated on a subsample for larger nodes |
| `min_share_prob` | `ρ` | per dataset | Minimum fraction of the parent's mass for every child |
| `use_item_mass` | `p_i` | per dataset | Item mass = training interaction frequency (`false` = uniform) |
| `milp_refine_rounds` | | 3 | Exact assignments per split, alternating with mass-weighted center updates |
| `milp_use_weighted_centers` | | true | Mass-weighted KMeans initialization and center updates |
| `weighted_silhouette` | | true | Score splits with the mass-weighted silhouette |
| `milp_candidate_topk` | | 16 | Each item may be assigned to one of its 16 nearest centers |
| `milp_time_limit`, `milp_gap` | | 100 s, 1e-3 | Per-(node, K) solver budget |
| `milp_solver_name` | | `CBC` | `CBC` (open source) or `GRB` (Gurobi) |

### Model (`conf/model/t5.yaml`) and training (`conf/training/default.yaml`)

| Key | Value | Key | Value |
|---|---|---|---|
| `encoder_layers` / `decoder_layers` | 6 / 6 | `learning_rate` | 1e-3 |
| `d_model` / `d_ff` / `num_heads` | 128 / 1024 / 6 | `weight_decay` | 1e-4 |
| `dropout_rate` | 0.15 | `per_device_train_batch_size` | 256 |
| `initializer_factor` | 0.02 | `bf16` | true |
| `layer_norm_epsilon` | 1e-6 | `early_stopping.patience` | 5 evaluations |
| `generation.num_beams` | 10 | `metric_for_best_model` | `ndcg@10` |

To log to Weights & Biases, pass `training.hf.report_to=wandb`.

---

## Citation

```bibtex
@inproceedings{hadad2026ast,
  title     = {Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation},
  author    = {Hadad, Guy and Roitman, Haggai and Shapira, Bracha},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  series    = {RecSys '26},
  pages     = {390--399},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3773078.3831756}
}
```

## License

MIT, see [LICENSE](LICENSE).
