"""CROPTrainer: the full CROP training procedure (Algorithm 1 of the paper).

  Phase 0   score every document with the frozen reference model (once),
            cache per-document mean token losses.
  Stage A   train on documents above the doc-gate quantile (default: drop the
            bottom 30% by mean score), full next-token loss, short context,
            WSD warmup -> stable.
  Stage B   replay upper score tiers with the rho(t)-masked loss; context
            ramps linearly ctx_start -> ctx_end; advance when the val
            excess-loss slope stalls below gamma (or the budget is spent).
  Stage C   anneal the LR on the top-quality slice; the Stage-B checkpoint is
            saved first as a reusable asset (compute recycling).

Everything is accounted: the FLOPs meter integrates the actual per-step cost
at the actual context length, plus reference-scoring overhead, and the run
summary reports FLOPs against the uniform-4K baseline (the live analogue of
the paper's Table 1).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import RunConfig
from ..data.corpus import (
    DocStore,
    QuantileSampler,
    StreamingPacker,
    make_synthetic_corpus,
    read_documents,
)
from ..data.scorer import (
    load_doc_scores,
    save_doc_scores,
    score_documents,
    summarize_scores,
)
from ..models import build_model, build_reference_model, logits_of
from ..optim.lr import WSDLRSchedule
from ..optim.muon import build_optimizer
from .flops import FLOPsMeter, format_flops, train_flops
from .gates import ExcessLossGate
from .masks import ReplayMaskScheduler, masked_ce_loss

STAGE_NAMES = ("A", "B", "C")


class CROPTrainer:
    def __init__(
        self,
        cfg: RunConfig,
        model: Optional[torch.nn.Module] = None,
        ref_model: Optional[torch.nn.Module] = None,
        verbose: bool = True,
        # re-anneal branch mode (compute recycling, Stage C only)
        load_model_from: Optional[str] = None,
        branch_budget_tokens: Optional[int] = None,
        branch_seed: int = 0,
        tag: str = "",
    ):
        self.cfg = cfg
        cfg.validate()
        if not cfg.crop.enabled:
            # B1-style uniform WSD baseline: disable gating/replay/gate
            cfg.crop.doc_drop_quantile = 0.0
            cfg.crop.stage_b_doc_quantile = 0.0
            cfg.crop.stage_c_doc_quantile = 0.0
            cfg.crop.rho_start = cfg.crop.rho_end = 0.0
            cfg.crop.gate_enabled = False
        self.verbose = verbose
        self.branch_mode = load_model_from is not None
        self.branch_seed = branch_seed
        self.tag = tag or ("branch" if self.branch_mode else "")

        self.device = self._resolve_device(cfg.engine.device)
        self._set_seeds(cfg.engine.seed if not self.branch_mode else branch_seed)

        os.makedirs(cfg.engine.out_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.engine.out_dir, "config.yaml"))

        # ------------------------------------------------ data & tokenizer
        self.tokenizer, self.train_store, self.val_store = self._prepare_data()

        # ------------------------------------------------ models
        self.model = model or build_model(cfg.model, vocab_size=self.tokenizer.vocab_size)
        self.model.to(self.device)
        self.ref_model = ref_model or build_reference_model(cfg.ref, vocab_size=self.tokenizer.vocab_size)
        self.ref_model.to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_ref_params = sum(p.numel() for p in self.ref_model.parameters())
        self.d_model, self.n_layers = self._arch_dims()

        # ------------------------------------------------ Phase 0: doc scores
        self.phase0_scoring_tokens = 0
        self.phase0_flops = 0.0
        self.doc_scores = self._ensure_doc_scores()

        # ------------------------------------------------ schedule & budgets
        sc = cfg.schedule
        if self.branch_mode:
            budgets = [0, 0, branch_budget_tokens or int(sc.total_tokens * sc.stage_shares[2])]
        else:
            budgets = list(cfg.stage_budgets())
        self.stage_budgets: List[int] = budgets
        self.total_budget = int(sc.total_tokens)

        self.sched = WSDLRSchedule(
            peak_lr=sc.peak_lr,
            end_lr_fraction=sc.end_lr_fraction,
            warmup_fraction=sc.warmup_fraction,
            total_tokens=self.total_budget,
        )
        self._sync_sched_boundaries()

        self.optimizer = build_optimizer(
            self.model, sc.optimizer, sc.peak_lr, sc.weight_decay,
            sc.muon_momentum, tuple(sc.adam_betas),
        )
        self.amp_dtype = self._resolve_amp(sc.amp_dtype)

        # ------------------------------------------------ meters & gate
        self.meter = FLOPsMeter(
            n_params=float(self.n_params),
            n_layers=self.n_layers,
            d_model=self.d_model,
            ref_n_params=float(self.n_ref_params),
            budget_tokens=self.total_budget,
        )
        self.gate = ExcessLossGate(
            gamma=cfg.crop.gate_gamma,
            window_points=cfg.engine.eval_batches + 1,
            patience=cfg.crop.gate_patience,
        )
        self.replay = ReplayMaskScheduler(
            cfg.crop.rho_start, cfg.crop.rho_end,
            stage_budget_tokens=max(budgets[1], 1),
        )

        # fixed evaluation batches (comparable across evals)
        self.eval_batches = self._make_eval_batches()

        # ------------------------------------------------ run state
        self.stage_idx = 2 if self.branch_mode else 0
        self.tokens_total = 0
        self.tokens_in_stage = 0
        self.opt_steps = 0
        self.stage_log: List[dict] = []
        self.best_val_excess: Optional[float] = None
        self._smooth_loss: Optional[float] = None
        self._t0 = time.time()
        self.log_path = os.path.join(cfg.engine.out_dir, "log.jsonl")

        if cfg.engine.resume and not self.branch_mode and os.path.exists(self._ckpt("last.pt")):
            if self.load_last():
                print(f"[resume] restored from {self._ckpt('last.pt')} at step {self.opt_steps}, "
                      f"stage {STAGE_NAMES[self.stage_idx]}, {self.tokens_total:,} tokens")

        if self.branch_mode:
            # compute recycling: load stable weights; fresh optimizer is kept
            ck = torch.load(load_model_from, map_location=self.device, weights_only=False)
            state = ck.get("model", ck)
            self.model.load_state_dict(state)
            if self.verbose:
                print(f"[branch] loaded stable weights from {load_model_from}")

    # =================================================================== #
    # setup helpers
    # =================================================================== #
    def _resolve_device(self, dev: str) -> str:
        if dev != "auto":
            return dev
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _set_seeds(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))

    def _resolve_amp(self, name: str):
        if name == "bf16":
            return torch.bfloat16
        if name == "fp16":
            return torch.float16
        return None

    def _arch_dims(self) -> Tuple[int, int]:
        m = self.model
        if hasattr(m, "cfg"):  # TinyGPT
            return int(m.cfg["d_model"]), len(m.blocks)
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "hidden_size"):
            return int(c.hidden_size), int(c.num_hidden_layers)
        return 0, 0

    def _prepare_data(self):
        d = self.cfg.data
        tok = None  # built lazily by corpus path
        if d.corpus == "synthetic":
            docs = make_synthetic_corpus(
                n_docs=d.synthetic_docs, doc_len=d.synthetic_doc_len,
                vocab_words=d.synthetic_vocab_words, seed=d.seed,
            )
        else:
            docs = read_documents(d.corpus)
        from ..tokenizer import build_tokenizer

        tok = build_tokenizer(d.tokenizer)
        store = DocStore.tokenize(docs, tok)
        if d.val_corpus:
            val_store = DocStore.tokenize(read_documents(d.val_corpus), tok)
        else:
            store, val_store = store.split_val(d.val_fraction, seed=d.seed)
        if self.verbose:
            print(
                f"[data] {len(store)} train docs / {store.n_tokens:,} tokens; "
                f"{len(val_store)} val docs; vocab={tok.vocab_size}"
            )
        return tok, store, val_store

    def _ensure_doc_scores(self) -> np.ndarray:
        d = self.cfg.data
        cache_dir = d.cache_dir or os.path.join(self.cfg.engine.out_dir, "cache")
        cached = load_doc_scores(cache_dir)
        if cached is not None and len(cached) == len(self.train_store):
            if self.verbose:
                print(f"[phase0] loaded cached doc scores ({len(cached)} docs)")
            return cached
        if self.verbose:
            print(f"[phase0] scoring {len(self.train_store)} docs with "
                  f"{self.n_ref_params/1e6:.1f}M-param reference model...")
        scores = score_documents(
            self.ref_model, self.train_store, device=self.device,
            batch_docs=self.cfg.ref.batch_size,
            max_docs=self.cfg.ref.score_max_docs,
            max_doc_tokens=self.ref_model.max_seq_len if hasattr(self.ref_model, "max_seq_len") else 0,
        )
        self.phase0_scoring_tokens = int(self.train_store.n_tokens)
        self.phase0_flops = 2.0 * self.n_ref_params * self.phase0_scoring_tokens
        save_doc_scores(
            cache_dir, scores,
            meta={"n_docs": len(scores), "tokenizer": d.tokenizer,
                  "ref_params": self.n_ref_params,
                  "summary": summarize_scores(scores)},
        )
        if self.verbose:
            s = summarize_scores(scores)
            print(f"[phase0] score mean={s['mean']:.3f} "
                  f"q30={s['quantiles_10_30_50_70_90'][1]:.3f}")
        return scores

    def _sampler_for_stage(self, stage: int) -> QuantileSampler:
        c = self.cfg.crop
        lo = {0: c.doc_drop_quantile, 1: c.stage_b_doc_quantile, 2: c.stage_c_doc_quantile}[stage]
        seed = (self.cfg.schedule.shuffle_seed + 1000 * stage
                + (self.branch_seed * 7919 if self.branch_mode else 0))
        return QuantileSampler(self.doc_scores, lo=lo, hi=1.0, seed=seed)

    def _make_eval_batches(self) -> List[torch.Tensor]:
        sampler = QuantileSampler(np.zeros(len(self.val_store)), lo=0.0, hi=1.0, seed=0)
        packer = StreamingPacker(self.val_store, sampler, seed=0)
        B, L = self.cfg.schedule.batch_size, self.cfg.schedule.context_end
        # keep eval memory small on huge batch sizes
        B = max(1, min(B, 16))
        return [
            torch.from_numpy(packer.next_batch(B, L)).to(self.device)
            for _ in range(self.cfg.engine.eval_batches)
        ]

    def _sync_sched_boundaries(self) -> None:
        a_end = self.stage_budgets[0]
        b_end = a_end + self.stage_budgets[1]
        if self.branch_mode:
            # fresh decay: boundaries relative to branch tokens
            self.sched.total_tokens = self.stage_budgets[2]
            self.sched._warmup_end = 0
            self.sched._decay_start = 0
            self.sched._decay_end = max(self.stage_budgets[2], 1)
        else:
            self.sched.set_stage_boundaries(a_end, b_end, self.total_budget)

    # =================================================================== #
    # main loop
    # =================================================================== #
    def train(self) -> Dict:
        if self.branch_mode:
            print(f"[branch:{self.tag}] re-anneal from {self.cfg.engine.out_dir} "
                  f"(budget {self.stage_budgets[2]:,} tokens, seed {self.branch_seed})")
        else:
            print(f"[crop] run: {self.n_params/1e6:.1f}M-param student, "
                  f"{self.n_ref_params/1e6:.1f}M ref, budget {self.total_budget:,} tokens "
                  f"@ {self.device}")
        start = self.stage_idx
        if (not self.branch_mode and self.tokens_in_stage == 0 and self.stage_log
                and self.stage_log[-1]["stage"] == STAGE_NAMES[self.stage_idx]):
            start = min(self.stage_idx + 1, 3)  # stage already completed
        for s in range(start, 3):
            if self.stage_budgets[s] <= 0:
                continue
            self._run_stage(s)
        return self._finish()

    def load_last(self, path: Optional[str] = None) -> bool:
        """Restore model/optimizer/schedule/meters from the rolling checkpoint.
        Note: the stream packer is re-seeded, so a mid-stage resume may re-see
        a small amount of data from the current window."""
        path = path or self._ckpt("last.pt")
        if not os.path.exists(path):
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["sched"])
        self.meter.load_state_dict(ck["meter"])
        self.gate.load_state_dict(ck["gate"])
        self.stage_idx = int(ck["stage_idx"])
        self.tokens_total = int(ck["tokens_total"])
        self.tokens_in_stage = int(ck["tokens_in_stage"])
        self.stage_budgets = list(ck["stage_budgets"])
        self.opt_steps = int(ck["opt_steps"])
        self.stage_log = list(ck.get("stage_log", []))
        return True

    def _ctx_for(self, stage: int) -> int:
        sc = self.cfg.schedule
        if stage == 0:
            return sc.context_start
        if stage == 1:
            budget = max(self.stage_budgets[1], 1)
            frac = min(self.tokens_in_stage / budget, 1.0)
            return int(round(sc.context_start + (sc.context_end - sc.context_start) * frac))
        return sc.context_end

    def _token_scores(self, batch: torch.Tensor) -> torch.Tensor:
        """Per-token reference loss s(x_i) for the batch, no grad [B, T]."""
        with torch.no_grad():
            ref_logits = logits_of(self.ref_model(batch)).float()
            ce = F.cross_entropy(
                ref_logits[:, :-1].reshape(-1, ref_logits.size(-1)),
                batch[:, 1:].reshape(-1),
                reduction="none",
            ).view(batch.size(0), -1)
            # pad last position with the previous score (targets align to [:, 1:])
            last = ce[:, -1:]
            return torch.cat([ce, last], dim=1)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float]:
        self.model.eval()
        ce_s, ce_r, n = 0.0, 0.0, 0
        for batch in self.eval_batches:
            logits = logits_of(self.model(batch)).float()
            tgt = batch[:, 1:]
            s = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tgt.reshape(-1),
                reduction="mean",
            ).item()
            ref_logits = logits_of(self.ref_model(batch)).float()
            r = F.cross_entropy(
                ref_logits[:, :-1].reshape(-1, ref_logits.size(-1)), tgt.reshape(-1),
                reduction="mean",
            ).item()
            ce_s += s * tgt.numel()
            ce_r += r * tgt.numel()
            n += tgt.numel()
        self.model.train()
        ce_s, ce_r = ce_s / n, ce_r / n
        return ce_s, ce_r, ce_s - ce_r

    def _run_stage(self, stage: int) -> None:
        cfg = self.cfg
        sc, cr = cfg.schedule, cfg.crop
        self.stage_idx = stage
        budget = self.stage_budgets[stage]
        sampler = self._sampler_for_stage(stage)
        packer = StreamingPacker(self.train_store, sampler,
                                 seed=cfg.schedule.shuffle_seed + stage)
        self.gate.reset_stage()
        if stage == 1:
            self.replay = ReplayMaskScheduler(cr.rho_start, cr.rho_end,
                                              stage_budget_tokens=max(budget, 1))
        name = STAGE_NAMES[stage]
        if self.verbose:
            extra = {
                0: f"doc gate: keep top {100*(1-cr.doc_drop_quantile):.0f}% by mean score",
                1: (f"replay upper tiers (score >= q{100*cr.stage_b_doc_quantile:.0f}), "
                    f"rho {cr.rho_start}->{cr.rho_end}, ctx {sc.context_start}->{sc.context_end}"),
                2: f"anneal on top {100*(1-cr.stage_c_doc_quantile):.0f}% slice",
            }[stage]
            print(f"[stage {name}] budget {budget:,} tokens | {extra}")

        self.model.train()
        eval_every_steps = max(cfg.engine.eval_every, 1)
        while self.tokens_in_stage < budget:
            ctx = self._ctx_for(stage)
            B = sc.batch_size
            micro = max(B // max(sc.grad_accum, 1), 1)
            self.optimizer.zero_grad(set_to_none=True)

            step_loss, step_tokens, step_masked = 0.0, 0, 0
            for _ in range(max(sc.grad_accum, 1)):
                batch_np = packer.next_batch(micro, ctx)
                batch = torch.from_numpy(batch_np).to(self.device)
                rho = self.replay.rho_at(self.tokens_in_stage) if stage == 1 else 0.0
                if stage == 1:
                    scores = self._token_scores(batch)
                    w_full = self.replay.mask_from_scores(scores, rho)
                    w = w_full[:, 1:]
                    self.meter.add_ref_forward(batch.numel(), ctx)
                else:
                    w = None

                with torch.autocast(self.device, dtype=self.amp_dtype,
                                    enabled=self.amp_dtype is not None):
                    logits = logits_of(self.model(batch[:, :-1]))
                    if w is None:
                        w = torch.ones_like(batch[:, 1:], dtype=torch.float32)
                    loss, _, kept = masked_ce_loss(logits, batch[:, 1:], w)
                    (loss / max(sc.grad_accum, 1)).backward()

                step_loss += loss.item()
                step_tokens += batch.numel()
                step_masked += (micro * (ctx - 1)) - kept

            gnorm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), sc.grad_clip)
            lr = self.sched.lr_at(self.tokens_total)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            self.optimizer.step()

            self.opt_steps += 1
            self.tokens_total += step_tokens
            self.tokens_in_stage += step_tokens
            self.meter.add_student_step(step_tokens, ctx, n_masked=step_masked)
            self._smooth_loss = (step_loss if self._smooth_loss is None
                                 else 0.9 * self._smooth_loss + 0.1 * step_loss)

            if self.opt_steps % cfg.engine.log_every == 0 and self.verbose:
                self._log_line(name, step_loss, lr, ctx, gnorm)
                self._jsonl(name, loss=step_loss, lr=lr, ctx=ctx)

            if self.opt_steps % eval_every_steps == 0 and stage in (0, 1) and cr.gate_enabled:
                s_ce, r_ce, excess = self.evaluate()
                self.gate.record(self.tokens_in_stage, excess)
                self.best_val_excess = excess if self.best_val_excess is None else min(self.best_val_excess, excess)
                if self.verbose:
                    print(f"  [eval {self.opt_steps:>6}] stage {name} student_ce={s_ce:.4f} "
                          f"ref_ce={r_ce:.4f} excess={excess:+.4f} "
                          f"({self.tokens_in_stage/budget:5.1%} of stage)")
                self._jsonl(name, val_student_ce=s_ce, val_ref_ce=r_ce,
                            val_excess=excess, lr=lr, ctx=ctx)
                fire, reason = self.gate.should_advance(
                    self.tokens_in_stage, budget, cr.min_stage_fraction)
                if fire:
                    print(f"  [gate] stage {name} -> advance ({reason})")
                    self._transition(stage, early=True)
                    return
                if cfg.engine.save_last_ckpt:
                    self._save_ckpt("last.pt")

        self._transition(stage, early=False)

    def _transition(self, stage: int, early: bool) -> None:
        consumed = self.tokens_in_stage
        self.stage_log.append({
            "stage": STAGE_NAMES[stage],
            "planned_tokens": self.stage_budgets[stage],
            "consumed_tokens": consumed,
            "cum_tokens_end": self.tokens_total,
            "early_gate": early,
            "ctx_end": self._ctx_for(stage),
        })
        self.stage_idx = stage
        leftover = max(self.stage_budgets[stage] - consumed, 0)
        if early and stage < 2:
            # donate unspent budget to the next stage (adaptive saving)
            self.stage_budgets[stage + 1] += leftover
            print(f"  [budget] {leftover:,} tokens donated to stage "
                  f"{STAGE_NAMES[stage + 1]}")
        self.tokens_in_stage = 0
        self._sync_sched_boundaries_after(stage)
        if stage == 1 and self.cfg.engine.save_stable_ckpt:
            self._save_ckpt("stable.pt", stable=True)
            print(f"  [ckpt] stable checkpoint saved -> {os.path.join(self.cfg.engine.out_dir, 'stable.pt')}")

    def _sync_sched_boundaries_after(self, stage: int) -> None:
        """Recompute WSD boundaries from *actual* stage ends (gate shifts them)."""
        if self.branch_mode:
            return
        ends = {s["stage"]: s["cum_tokens_end"] for s in self.stage_log}
        a_end = ends.get("A", 0)
        b_end = ends.get("B", a_end)
        self.sched.set_stage_boundaries(a_end, b_end, self.total_budget)

    # =================================================================== #
    # checkpoints & logging
    # =================================================================== #
    def _ckpt(self, name: str) -> str:
        return os.path.join(self.cfg.engine.out_dir, name)

    def _save_ckpt(self, name: str, stable: bool = False) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "sched": self.sched.state_dict(),
                "meter": self.meter.state_dict(),
                "gate": self.gate.state_dict(),
                "stage_idx": self.stage_idx,
                "tokens_total": self.tokens_total,
                "tokens_in_stage": self.tokens_in_stage,
                "stage_budgets": self.stage_budgets,
                "opt_steps": self.opt_steps,
                "stage_log": self.stage_log,
                "stable": stable,
            },
            self._ckpt(name),
        )

    def _log_line(self, stage: str, loss: float, lr: float, ctx: int, gnorm) -> None:
        m = self.meter
        print(
            f"  [step {self.opt_steps:>6}] {stage} loss={loss:.4f} (ema {self._smooth_loss:.4f}) "
            f"lr={lr:.2e} ctx={ctx} gnorm={float(gnorm):.2f} "
            f"tok={m.tokens_trained/1e6:,.2f}M flops={format_flops(m.flops_total)}"
        )

    def _jsonl(self, stage: str, **kw) -> None:
        rec = {
            "t": round(time.time() - self._t0, 2),
            "step": self.opt_steps,
            "stage": stage,
            "tokens_total": self.tokens_total,
            "tokens_in_stage": self.tokens_in_stage,
            "flops_total": self.meter.flops_total,
            "flops_student": self.meter.flops_student,
            "flops_ref": self.meter.flops_ref,
            "masked_fraction": self.meter.tokens_masked / max(self.meter.tokens_trained, 1),
            "ema_loss": self._smooth_loss,
        }
        rec.update(kw)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # =================================================================== #
    # summary
    # =================================================================== #
    def _finish(self) -> Dict:
        self._save_ckpt("final.pt")
        s_ce, r_ce, excess = self.evaluate()
        m = self.meter
        baseline = train_flops(
            self.total_budget, float(self.n_params), self.n_layers,
            self.d_model, self.cfg.schedule.context_end,
        )
        proj = self._paper_projection()
        summary = {
            "tag": self.tag,
            "branch_mode": self.branch_mode,
            "model_params": self.n_params,
            "ref_params": self.n_ref_params,
            "stage_log": self.stage_log,
            "tokens_total": self.tokens_total,
            "phase0": {
                "scoring_tokens": self.phase0_scoring_tokens,
                "flops": self.phase0_flops,
            },
            "flops": m.summary(),
            "baseline_flops_uniform_ctx_end": baseline,
            "flops_vs_baseline": m.flops_total / baseline,
            "val": {"student_ce": s_ce, "ref_ce": r_ce, "excess": excess},
            "wall_clock_s": round(time.time() - self._t0, 1),
            "paper_projection": proj,
        }
        with open(os.path.join(self.cfg.engine.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if self.verbose:
            print(
                f"[done] tokens={self.tokens_total:,} "
                f"flops={format_flops(m.flops_total)} "
                f"({100 * summary['flops_vs_baseline']:.1f}% of uniform baseline) "
                f"val_ce={s_ce:.4f} excess={excess:+.4f}"
            )
        return summary

    def _paper_projection(self) -> Dict[str, float]:
        """Table-1-style projection using this run's actual sizes."""
        from .flops import project_crop

        c = self.cfg
        return project_crop(
            n_params=float(self.n_params),
            d_model=self.d_model or 64,
            n_layers=self.n_layers or 4,
            budget_tokens=float(c.schedule.total_tokens),
            stage_shares=tuple(c.schedule.stage_shares),
            ctx_start=c.schedule.context_start,
            ctx_end=c.schedule.context_end,
            doc_drop=c.crop.doc_drop_quantile,
            n_ref=float(self.n_ref_params),
            corpus_inflation=c.ref.corpus_inflation,
        )
