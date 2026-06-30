import torch

from quac.training.stargan import build_model
from quac.training.solver import (
    PretrainSolver,
    compute_pretrain_d_loss,
    compute_pretrain_g_loss,
    kl_divergence,
)


def _variational_nets():
    return build_model(
        img_size=64,
        style_dim=64,
        latent_dim=16,
        num_domains=1,
        input_dim=1,
        variational=True,
    )


def test_kl_divergence_zero_at_standard_normal():
    # KL(N(0, I) || N(0, I)) == 0 (mu = 0, logvar = 0).
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    assert torch.allclose(kl_divergence(mu, logvar), torch.tensor(0.0), atol=1e-6)
    # Non-trivial posterior has positive KL.
    assert kl_divergence(torch.ones(4, 8), torch.zeros(4, 8)) > 0


def test_compute_pretrain_d_loss():
    nets, _ = _variational_nets()
    x = torch.randn(2, 1, 64, 64)
    y = torch.zeros(2, dtype=torch.long)
    loss, parts = compute_pretrain_d_loss(nets, x, y)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert set(parts) == {"real", "fake", "reg"}


def test_compute_pretrain_g_loss():
    nets, _ = _variational_nets()
    x = torch.randn(2, 1, 64, 64)
    y = torch.zeros(2, dtype=torch.long)
    loss, parts, x_fake, x_rec = compute_pretrain_g_loss(nets, x, y)
    assert torch.isfinite(loss)
    assert x_fake.shape == x.shape
    assert x_rec.shape == x.shape
    assert set(parts) == {"adv", "recon", "kl_c", "kl_s", "ds"}


class DummyLoader:
    """Minimal stand-in for TrainingData: yields a constant batch dict."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __next__(self):
        return {
            "x_src": self.x,
            "x_src2": self.x,
            "y_src": self.y,
            "x_ref": self.x,
            "x_ref2": self.x,
            "y_ref": self.y,
        }


def test_pretrain_solver_runs(tmp_path):
    nets, nets_ema = _variational_nets()
    solver = PretrainSolver(
        nets,
        nets_ema,
        f_lr=1e-4,
        lr=1e-4,
        beta1=0.0,
        beta2=0.99,
        weight_decay=1e-4,
        root_dir=str(tmp_path),
        run=None,
    )
    x = torch.randn(2, 1, 64, 64)
    y = torch.zeros(2, dtype=torch.long)
    loader = DummyLoader(x, y)
    solver.train(
        loader,
        total_iters=2,
        log_every=1,
        save_every=2,
        ds_iter=2,
        kl_anneal_iters=2,
    )
    # A checkpoint was written at the final step (Solver saves the EMA nets).
    assert list((tmp_path / "checkpoints").glob("*_nets_ema.ckpt"))
