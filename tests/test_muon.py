"""Muon optimizer sanity: orthogonalized-momentum updates must minimize a
small least-squares problem, and the factory must wire Muon+AdamW groups."""

import torch

from shimballm.optim.muon import Muon, build_optimizer, zeropower_via_newtonschulz5


def test_newtonschulz_orthogonalizes():
    """NS5 is an *approximate* orthogonalization (by design, matching the
    official Muon); assert it lands near semi-orthogonal and strictly
    improves on the raw matrix."""
    torch.manual_seed(0)
    A = torch.randn(16, 8)
    A_hat = A / (A.norm() + 1e-7)
    O = zeropower_via_newtonschulz5(A, steps=5).float()
    dev = lambda M: (M.mT @ M - torch.eye(8)).abs().max().item()
    assert dev(O) < 0.35
    assert dev(O) < 0.5 * dev(A_hat)  # strictly improves orthogonality


def test_muon_minimizes_least_squares():
    torch.manual_seed(0)
    W = torch.randn(8, 4)  # unknown target
    x = torch.randn(64, 8)
    y = x @ W + 0.01 * torch.randn(64, 4)

    model = torch.nn.Linear(8, 4, bias=False)
    opt = Muon(muon_params=[model.weight], lr=0.05)
    first = None
    for i in range(60):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_factory_groups():
    from shimballm.models.tiny_gpt import TinyGPT

    m = TinyGPT(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=32)
    opt = build_optimizer(m, "muon", peak_lr=1e-3, weight_decay=0.1,
                          muon_momentum=0.95, adam_betas=(0.9, 0.95))
    muon_groups = [g for g in opt.param_groups if g.get("use_muon")]
    adam_groups = [g for g in opt.param_groups if not g.get("use_muon")]
    assert muon_groups and adam_groups
    assert all(p.ndim == 2 for g in muon_groups for p in g["params"])
    assert all(p.ndim != 2 for g in adam_groups for p in g["params"])
    # one step runs
    x = torch.randint(0, 32, (2, 8))
    loss = m(x).sum()
    loss.backward()
    opt.step()
