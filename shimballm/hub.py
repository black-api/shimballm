"""Hugging Face Hub integration for the ShimbaLLM org.

Pushes trained checkpoints as proper HF model repos: config.json, weights,
generation-less model card with the CROP recipe summary, and tags. Works for
both built-in TinyGPT checkpoints and HF ``PreTrainedModel`` students.

    from shimballm.hub import push_model
    push_model("runs/demo/final.pt", cfg, repo_id="ShimbaLLM/crop-tiny-demo")
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .config import RunConfig

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
library_name: shimballm
pipeline_tag: text-generation
tags:
- shimballm
- crop
- small-language-model
- compute-efficient-training
- selective-token-replay
- wsd-schedule
---

# {title}

Trained with **ShimbaLLM** (CROP: Curriculum-guided Reduction Of Pretraining) --
loss-gated data staging with selective token replay for compute-efficient
small language model training.

## Recipe (CROP)

| Component | Setting |
|---|---|
| Student | {n_params_millions:.1f}M params |
| Reference scorer | {n_ref_millions:.1f}M params (frozen, excess loss) |
| Doc gate theta_doc | drop bottom {doc_drop_pct:.0f}% by mean token score |
| Stage B replay | mask easiest tokens, rho {rho_start:.2f} -> {rho_end:.2f} |
| Context ramp | {ctx_start} -> {ctx_end} tokens |
| Schedule | WSD, shares A/B/C {share_a:.0f}/{share_b:.0f}/{share_c:.0f} |
| Token budget | {total_tokens:,} |

## Results (this run)

- validation CE: {student_ce:.4f} (reference CE {ref_ce:.4f}, excess {excess:+.4f})
- training FLOPs: {flops:.3e} = {flops_pct:.1f}% of the uniform baseline
- re-anneal branches: {n_branches}

## Usage

```python
from shimballm import RunConfig
from shimballm.models import build_model
import torch

cfg = RunConfig.from_yaml("config.yaml")
model = build_model(cfg.model, vocab_size={vocab_size})
sd = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(sd)
```

See https://huggingface.co/{org} for more ShimbaLLM models.
"""


def make_model_card(cfg: RunConfig, summary: Optional[dict] = None) -> str:
    s = summary or {}
    val = s.get("val", {})
    m = cfg.model
    return MODEL_CARD_TEMPLATE.format(
        title=(cfg.hub.repo_name or "shimballm-crop-model").replace("-", " ").title(),
        n_params_millions=s.get("model_params", 0) / 1e6 or _cfg_params_millions(cfg),
        n_ref_millions=s.get("ref_params", cfg.ref.params_target) / 1e6,
        doc_drop_pct=100 * cfg.crop.doc_drop_quantile,
        rho_start=cfg.crop.rho_start,
        rho_end=cfg.crop.rho_end,
        ctx_start=cfg.schedule.context_start,
        ctx_end=cfg.schedule.context_end,
        share_a=cfg.schedule.stage_shares[0],
        share_b=cfg.schedule.stage_shares[1],
        share_c=cfg.schedule.stage_shares[2],
        total_tokens=cfg.schedule.total_tokens,
        student_ce=val.get("student_ce", float("nan")),
        ref_ce=val.get("ref_ce", float("nan")),
        excess=val.get("excess", float("nan")),
        flops=s.get("flops", {}).get("flops_total", float("nan")),
        flops_pct=100 * s.get("flops_vs_baseline", float("nan")),
        n_branches=len(s.get("branches", [])),
        org=cfg.hub.org,
        vocab_size=m.vocab_size,
    )


def _cfg_params_millions(cfg: RunConfig) -> float:
    try:
        from .models import build_model  # local import to avoid cycles

        return build_model(cfg.model, vocab_size=cfg.model.vocab_size).num_params() / 1e6
    except Exception:
        return float("nan")


def push_model(
    ckpt_path: str,
    cfg: RunConfig,
    repo_id: Optional[str] = None,
    summary_path: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload a trained checkpoint to the HF Hub (default org: ShimbaLLM).

    Creates ``<repo_id>`` with: config.yaml, pytorch_model.bin, README.md
    (auto model card with the CROP recipe + results), and summary.json when
    available. For HF students, prefer ``save_pretrained`` + ``push_to_hub``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to push models. "
            "Install with: pip install 'shimballm[hf]'"
        ) from e

    repo_id = repo_id or cfg.repo_id()
    out_dir = os.path.dirname(os.path.abspath(ckpt_path))
    stage_dir = os.path.join(out_dir, "_hf_upload")
    os.makedirs(stage_dir, exist_ok=True)

    # weights
    import shutil

    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    torch.save(state, os.path.join(stage_dir, "pytorch_model.bin"))

    # config + card + summary
    cfg.save(os.path.join(stage_dir, "config.yaml"))
    summary = None
    sp = summary_path or os.path.join(out_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            summary = json.load(f)
        shutil.copy(sp, os.path.join(stage_dir, "summary.json"))
    with open(os.path.join(stage_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(make_model_card(cfg, summary))

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=stage_dir, repo_id=repo_id,
                      commit_message="ShimbaLLM CROP training run")
    return f"https://huggingface.co/{repo_id}"
