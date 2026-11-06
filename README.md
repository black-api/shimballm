# ShimbaLLM

**Compute-efficient training for small language models.**
ShimbaLLM implements **CROP** (Curriculum-guided Reduction Of Pretraining):
loss-gated data staging with selective token replay, a 2K→4K context ramp,
and a WSD backbone with compute recycling — coordinated in one adaptive,
FLOPs-accounted training engine.

> Based on the research paper *"Smaller, Faster, Smarter: Loss-Gated Data
> Staging with Selective Token Replay for Compute-Efficient Small Language
> Model Training"* (see [`docs/CROP_paper.pdf`](docs/CROP_paper.pdf)).

---

## Why

At fixed parameter count, recent small language models differ by **tens of
MMLU points** purely because of recipe choices (Phi-3-mini 69.0 vs
Gemma-2-2B 51.3 at ~3B). CROP turns those choices into a single procedure
whose compute bill is explicit and falsifiable:

| Component | Mechanism | FLOPs vs. uniform baseline (paper Table 1, 1.5B / 100B tokens) |
|---|---|---|
| Uniform baseline | all tokens, 4K context, full loss | 100% |
| **Document gating** | drop lowest-excess-loss 30% of documents — never forwarded or back-propagated | **−15.4%** |
| **Context ramp** | Stage A at 2K, Stage B ramps 2K→4K | **−6.6%** |
| **Token replay masking** | mask easiest tokens (no FLOP change; shifts the loss-vs-FLOPs curve down) | 0% |
| **Reference scoring** | 0.3B frozen scorer, forward-only over 1.4× corpus | +8.1% |
| **CROP (conservative)** | matched token budget | **88.0%** |
| **CROP (H2 scenario)** | matched validation loss, 20% effective-budget shift | **72.1%** |

Two properties make this more than a bag of tricks:

1. **Every stage transition is gated by a measured signal** (validation
   excess-loss slope), not a fixed token count — a noisy corpus advances to
   selectivity sooner; a rich corpus earns more stable compute.
2. **The run is re-usable.** The WSD schedule puts the whole decay inside
   Stage C, so the Stage-B stable checkpoint can be re-annealed on different
   high-quality slices — each additional compute budget costs ~15% of a full
   run.

## The pipeline

```
            Phase 0 (offline, once)          frozen ref model (0.3B)
 corpus ────────────────────────────────► per-token scores s(x_i)
                                              │  doc score = mean s
                                              ▼
 ┌─────────────────────── Stage A (55%) ───────────────────────┐
 │ docs with mean s ≥ q30   ·  full loss  ·  ctx 2K  ·  WSD warmup→stable │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼  advance when val excess-loss stalls (γ)
 ┌─────────────────────── Stage B (30%) ───────────────────────┐
 │ upper score tiers  ·  masked loss, ρ(t): 0.40→0.10  ·  ctx ramp 2K→4K  │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼  save stable.pt (reusable asset)
 ┌─────────────────────── Stage C (15%) ───────────────────────┐
 │ top 20% slice  ·  LR annealing  ·  final model              │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
                    re-anneal branches (compute recycling)
                    branch_1, branch_2, ... @ ~15% cost each
```

## Install

```bash
git clone https://github.com/ShimbaLLM/shimballm.git
cd shimballm
pip install -e .                # core: torch + numpy + pyyaml
pip install -e ".[hf]"          # optional: transformers / datasets / hub
```

or straight from GitHub, without cloning:

```bash
pip install git+https://github.com/ShimbaLLM/shimballm.git
```

Python ≥ 3.9, PyTorch ≥ 2.1. CPU works out of the box for small runs.

## Train on Google Colab (no local GPU needed)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShimbaLLM/shimballm/blob/main/notebooks/ShimbaLLM_CROP_Colab.ipynb)

`notebooks/ShimbaLLM_CROP_Colab.ipynb` runs the whole recipe in the cloud on a
free T4. It auto-detects the hardware and picks a track:

| Track | Hardware | What runs | Time |
|---|---|---|---|
| CPU smoke | no GPU | `tiny_cpu.yaml` — synthetic corpus, full pipeline | ~2 min |
| Real run | T4/L4/A100 | `smollm_135m_hf.yaml` — SmolLM2-135M on a streamed FineWeb-Edu sample | ~1–2 h at the default 30M-token budget |

The notebook patches the config for Colab automatically (fp16 on a T4, which
lacks bf16; micro-batch 2 × 4K ctx with grad-accum 32 to fit 16 GB VRAM),
projects the FLOPs bill before training, trains with `--resume` so a dropped
session continues from `last.pt`, runs two re-anneal branches, plots the
curves from `log.jsonl`, and finally pushes the model to the HF Hub with
`shimballm push`.

## Quickstart (60 seconds, CPU)

```bash
shimballm demo
```

runs the entire recipe on a synthetic corpus: Phase 0 scoring → Stages A/B/C
→ stable checkpoint → a re-anneal branch → and prints the FLOPs projection.
Expect `val_ce` to fall from ~4.5 to ~0.6 and the re-anneal branch to report
~15% of the baseline cost.

## Using the engine

### 1. Project your compute bill first

```bash
shimballm project                       # paper defaults: 1.5B student, 100B tokens
shimballm project --config examples/student_1p5b_crop.yaml
```

### 2. Score your corpus (Phase 0, cached)

```bash
shimballm score --config examples/tiny_cpu.yaml
```

Writes `<cache_dir>/doc_scores.npy` — per-document mean token loss under the
frozen reference model. Payed once per corpus; every later run reloads it.

### 3. Train

```bash
shimballm train --config examples/tiny_cpu.yaml          # CPU demo
shimballm train --config examples/smollm_135m_hf.yaml    # real 135M run (needs [hf] extra)
shimballm train --config my.yaml --resume                # resume from last.pt
```

Artifacts in `engine.out_dir`:

| File | Contents |
|---|---|
| `final.pt` | trained student (state dict + optimizer + meters) |
| `stable.pt` | **Stage-B stable checkpoint** — the recycling asset |
| `summary.json` | stage budgets, FLOPs accounting, val losses, projection |
| `log.jsonl` | per-step metrics (loss, lr, ctx, masked fraction, FLOPs) |
| `cache/doc_scores.npy` | Phase 0 document scores |
| `branches/<tag>/branch.pt` | re-anneal branch models |

### 4. Recycle: re-anneal branches

```bash
shimballm anneal --config examples/tiny_cpu.yaml --seed 1
shimballm anneal --config examples/tiny_cpu.yaml --seed 2 --budget 90000
```

or programmatically:

```python
from shimballm import RunConfig
from shimballm.engine.recycling import AnnealBranchManager

cfg = RunConfig.from_yaml("examples/tiny_cpu.yaml")
mgr = AnnealBranchManager(cfg)                 # loads runs/.../stable.pt
mgr.run_branch(branch_seed=1, tag="code-heavy")
mgr.run_branch(branch_seed=2, tag="math-heavy")
print(mgr.best())                              # pick by val CE
```

### 5. Push to the Hugging Face Hub (org `ShimbaLLM`)

```bash
export HF_TOKEN=...
shimballm push --config examples/tiny_cpu.yaml --repo ShimbaLLM/crop-tiny-demo
```

Uploads weights + `config.yaml` + `summary.json` and generates a model card
with the full CROP recipe table and this run's measured results.

### Library API

```python
from shimballm import RunConfig, CROPTrainer, project_crop

cfg = RunConfig.from_yaml("my.yaml")
trainer = CROPTrainer(cfg)
summary = trainer.train()
print(summary["val"], summary["flops_vs_baseline"])
```

## Configuration reference

| Section | Key knobs | Paper anchor |
|---|---|---|
| `crop` | `doc_drop_quantile` (θ_doc, default 0.30), `gate_gamma` (γ) — **the only two new hyper-parameters** | Eq. 3, Eq. 6 |
| `crop` | `rho_start`→`rho_end` (0.40→0.10): easiest-token share masked in Stage B, linear decay | Eq. 5 |
| `crop` | `stage_b_doc_quantile` / `stage_c_doc_quantile`: replay tiers / anneal slice | §4.4–4.5 |
| `schedule` | `stage_shares` (0.55/0.30/0.15), `context_start`→`context_end` (2K→4K), WSD `warmup_fraction`, `end_lr_fraction` | §4.1, §4.6 |
| `ref` | `arch`, size (keep ~5× smaller than student), `corpus_inflation` (1.4) | §4.2 |
| `engine` | `out_dir`, eval cadence, `resume`, `save_stable_ckpt` | — |
| `hub` | `org` (default `ShimbaLLM`), `repo_name`, `push_after_train` | — |

**Baselines for ablations** (paper protocol §6.1): B1 = uniform WSD — set
`crop.enabled: false`, `context_start == context_end`. B2 = B1 + doc gating
only (`rho_start=rho_end=0`). B3 = B1 + token masking only (drop quantile 0).
B4 = cosine schedule (`WSDLRSchedule.lr_at_cosine` is provided).

## Repository layout

```
shimballm/
├── notebooks/
│   └── ShimbaLLM_CROP_Colab.ipynb  # Colab: cloud training, no local GPU
├── config.py              # RunConfig + YAML (model/ref/data/crop/schedule/engine/hub)
├── tokenizer.py           # char tokenizer + HF adapter
├── data/
│   ├── corpus.py          # DocStore, QuantileSampler (tier curriculum), StreamingPacker
│   └── scorer.py          # Phase 0: excess-loss document scoring (cached)
├── models/
│   ├── tiny_gpt.py        # built-in decoder-only LM (RoPE, GQA, SwiGLU opt)
│   └── __init__.py        # build_model / build_reference_model (HF bridge)
├── optim/
│   ├── lr.py              # WSD schedule over token milestones
│   └── muon.py            # Muon (orthogonalized momentum) + AdamW factory
├── engine/
│   ├── flops.py           # 6ND + attention term; Table-1 projector; live meter
│   ├── masks.py           # Eq. 5 masked loss + rho(t) scheduler
│   ├── gates.py           # Eq. 6 adaptive stage gate (val excess-loss slope)
│   ├── recycling.py       # AnnealBranchManager (Stage-C fan-out)
│   └── trainer.py         # CROPTrainer: the full Algorithm 1
├── hub.py                 # HF Hub push + model card generation
└── cli.py                 # shimballm demo|project|score|train|anneal|push
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 34 tests: paper numbers, WSD, masks, gate, curriculum, smoke run
```

`tests/test_flops.py` reproduces the paper's Table 1 from the engine's
formulas — the accounting stays falsifiable as the code evolves.

## Notes & limitations

- Token replay masks **do not** save raw FLOPs (masked tokens still flow
  through the network); they buy learning efficiency per FLOP. The compute
  saving comes from document gating and the context ramp — exactly as in the
  paper.
- The projection numbers are analytic (6ND + attention term), not measured
  wall-clock; systems effects (memory, I/O, kernels) are out of scope.
- Reference-model bias is the main risk (a scorer that under-rates a domain
  starves it). Probe with mismatched-domain scorers before production runs;
  consider per-domain gate floors.
- Mid-stage resume re-seeds the stream packer (a small data overlap window).

## Citation

```bibtex
@techreport{shimballm2025crop,
  title  = {Smaller, Faster, Smarter: Loss-Gated Data Staging with Selective
            Token Replay for Compute-Efficient Small Language Model Training},
  author = {{ShimbaLLM}},
  year   = {2025},
  note   = {See docs/CROP\_paper.pdf; engine: github.com/ShimbaLLM/shimballm}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
