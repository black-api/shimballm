"""ShimbaLLM command line interface.

    shimballm project  [--config c.yaml]         # paper Table-1 projection
    shimballm score    [--config c.yaml]         # Phase 0 doc scoring (cached)
    shimballm train    [--config c.yaml]         # full CROP run (stages A-C)
    shimballm anneal   --config c.yaml --seed 1  # re-anneal branch (recycling)
    shimballm push     --config c.yaml --repo ShimbaLLM/my-model
    shimballm demo                               # tiny CPU end-to-end demo
"""

from __future__ import annotations

import argparse
import sys


def _load_config(args) -> "RunConfig":
    from .config import RunConfig

    if not getattr(args, "config", None):
        raise SystemExit("a --config YAML file is required (see examples/)")
    return RunConfig.from_yaml(args.config)


def cmd_project(args) -> int:
    from .engine.flops import project_crop, render_projection

    if getattr(args, "config", None):
        cfg = _load_config(args)
        p = project_crop(
            n_params=1.5e9 if args.paper else _param_count(cfg),
            d_model=2048 if args.paper else _d_model(cfg),
            n_layers=28 if args.paper else _n_layers(cfg),
            budget_tokens=float(cfg.schedule.total_tokens),
            stage_shares=tuple(cfg.schedule.stage_shares),
            ctx_start=cfg.schedule.context_start,
            ctx_end=cfg.schedule.context_end,
            doc_drop=cfg.crop.doc_drop_quantile,
            n_ref=float(cfg.ref.params_target),
            corpus_inflation=cfg.ref.corpus_inflation,
        )
    else:
        p = project_crop()  # exact paper defaults
    print(render_projection(p))
    return 0


def _param_count(cfg) -> float:
    # analytic estimate -- never instantiate large models just to count them
    return float(cfg.model.n_params_estimate())


def _d_model(cfg) -> int:
    if cfg.model.arch == "tiny":
        return cfg.model.d_model
    return int(getattr(cfg.model, "d_model", 2048))


def _n_layers(cfg) -> int:
    if cfg.model.arch == "tiny":
        return cfg.model.n_layers
    return int(getattr(cfg.model, "n_layers", 28))


def cmd_score(args) -> int:
    from .data.corpus import DocStore, make_synthetic_corpus, read_documents
    from .data.scorer import load_doc_scores, save_doc_scores, score_documents, summarize_scores
    from .models import build_reference_model
    from .tokenizer import build_tokenizer

    cfg = _load_config(args)
    d = cfg.data
    if d.corpus == "synthetic":
        docs = make_synthetic_corpus(d.synthetic_docs, d.synthetic_doc_len,
                                     d.synthetic_vocab_words, d.seed)
    else:
        docs = read_documents(d.corpus)
    tok = build_tokenizer(d.tokenizer)
    store = DocStore.tokenize(docs, tok)
    cache_dir = d.cache_dir or (cfg.engine.out_dir + "/cache")
    cached = load_doc_scores(cache_dir)
    if cached is not None and len(cached) == len(store) and not args.force:
        print(f"[score] cached scores found ({len(cached)} docs) -- nothing to do")
        return 0
    ref = build_reference_model(cfg.ref, vocab_size=tok.vocab_size)
    import torch

    device = cfg.engine.device if cfg.engine.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(ref, "max_seq_len"):
        max_doc_tokens = ref.max_seq_len
    else:  # HF: respect the positional-embedding budget
        max_doc_tokens = int(getattr(getattr(ref, "config", None),
                                     "max_position_embeddings", 0) or 0)
    scores = score_documents(ref, store, device=device, batch_docs=cfg.ref.batch_size,
                             max_docs=cfg.ref.score_max_docs,
                             max_doc_tokens=max_doc_tokens)
    path = save_doc_scores(cache_dir, scores, {"n_docs": len(scores),
                                               "tokenizer": d.tokenizer,
                                               "summary": summarize_scores(scores)})
    print(f"[score] saved {len(scores)} doc scores -> {path}")
    print(summarize_scores(scores))
    return 0


def cmd_train(args) -> int:
    from .engine.trainer import CROPTrainer

    cfg = _load_config(args)
    cfg.engine.resume = bool(args.resume) or cfg.engine.resume
    trainer = CROPTrainer(cfg)
    summary = trainer.train()
    if cfg.hub.push_after_train:
        from .hub import push_model

        url = push_model(f"{cfg.engine.out_dir}/final.pt", cfg,
                         private=cfg.hub.private)
        print(f"[hub] pushed -> {url}")
    print(f"[done] val_ce={summary['val']['student_ce']:.4f} "
          f"flops={summary['flops']['flops_total']:.3e} "
          f"({100 * summary['flops_vs_baseline']:.1f}% of uniform baseline)")
    return 0


def cmd_anneal(args) -> int:
    from .engine.recycling import AnnealBranchManager

    cfg = _load_config(args)
    mgr = AnnealBranchManager(cfg, stable_ckpt=args.stable)
    summary = mgr.run_branch(branch_seed=args.seed, budget_tokens=args.budget,
                             tag=getattr(args, "tag", None))
    print(f"[anneal] branch seed={args.seed} val_ce={summary['val']['student_ce']:.4f}")
    return 0


def cmd_push(args) -> int:
    from .hub import push_model

    cfg = _load_config(args)
    repo = args.repo or cfg.repo_id()
    ckpt = args.ckpt or f"{cfg.engine.out_dir}/final.pt"
    url = push_model(ckpt, cfg, repo_id=repo, private=args.private)
    print(f"[hub] pushed -> {url}")
    return 0


def cmd_demo(args) -> int:
    """Tiny end-to-end CROP run on CPU (~1 minute): synthetic corpus, ~0.3M
    student, ~0.05M reference, all three stages plus a re-anneal branch."""
    import os
    import tempfile

    from .engine.recycling import AnnealBranchManager
    from .engine.trainer import CROPTrainer

    tmp = args.out or os.path.join(tempfile.gettempdir(), "shimballm_demo")
    cfg_yaml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "examples", "tiny_cpu.yaml")
    cfg = _load_config(argparse.Namespace(config=cfg_yaml))
    cfg.engine.out_dir = tmp
    print(f"[demo] output -> {tmp}\n")
    trainer = CROPTrainer(cfg)
    summary = trainer.train()
    print(f"\n[demo] CROP run done: val_ce={summary['val']['student_ce']:.4f}, "
          f"FLOPs {100 * summary['flops_vs_baseline']:.1f}% of uniform baseline")
    mgr = AnnealBranchManager(cfg)
    branch = mgr.run_branch(branch_seed=1, tag="demo-branch")
    print(f"[demo] re-anneal branch: val_ce={branch['val']['student_ce']:.4f} "
          f"(best of {len(mgr.results)})")
    from .engine.flops import render_projection

    print("\n" + render_projection(summary["paper_projection"]))
    print(f"\n[demo] artifacts in {tmp} (final.pt, stable.pt, summary.json, log.jsonl)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="shimballm",
                                description="ShimbaLLM: CROP training engine for small language models")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("project", help="print the CROP FLOPs projection (paper Table 1)")
    sp.add_argument("--config", default=None, help="use sizes/budget from a config")
    sp.add_argument("--paper", action="store_true", help="force the paper's 1.5B/100B defaults")
    sp.set_defaults(fn=cmd_project)

    sp = sub.add_parser("score", help="Phase 0: reference-score the corpus (doc-level)")
    sp.add_argument("--config", required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_score)

    sp = sub.add_parser("train", help="run the full CROP procedure")
    sp.add_argument("--config", required=True)
    sp.add_argument("--resume", action="store_true")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("anneal", help="re-anneal a branch from the Stage-B stable checkpoint")
    sp.add_argument("--config", required=True)
    sp.add_argument("--stable", default=None)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--budget", type=int, default=None, help="branch token budget")
    sp.add_argument("--tag", default=None, help="branch tag, e.g. code-heavy")
    sp.set_defaults(fn=cmd_anneal)

    sp = sub.add_parser("push", help="push a checkpoint to the HF Hub (org ShimbaLLM)")
    sp.add_argument("--config", required=True)
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--repo", default=None, help="repo id, e.g. ShimbaLLM/crop-tiny")
    sp.add_argument("--private", action="store_true")
    sp.set_defaults(fn=cmd_push)

    sp = sub.add_parser("demo", help="tiny CPU end-to-end demo (score -> A -> B -> C -> branch)")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
