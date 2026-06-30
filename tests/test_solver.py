import torch

from quac.training.stargan import build_model
from quac.training.solver import Solver


class DummyLoader:
    """Minimal stand-in for TrainingData with distinct source/target domains."""

    def __init__(self, x, y_src, y_ref):
        self.x, self.y_src, self.y_ref = x, y_src, y_ref

    def __next__(self):
        return {
            "x_src": self.x,
            "x_src2": self.x,
            "y_src": self.y_src,
            "x_ref": self.x,
            "x_ref2": self.x,
            "y_ref": self.y_ref,
        }


def _build_solver(tmp_path):
    nets, nets_ema = build_model(
        img_size=64,
        style_dim=64,
        latent_dim=16,
        num_domains=2,
        input_dim=1,
    )
    return Solver(
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


def test_solver_runs_and_resumes(tmp_path):
    x = torch.randn(2, 1, 64, 64)
    y_src = torch.zeros(2, dtype=torch.long)
    y_ref = torch.ones(2, dtype=torch.long)
    loader = DummyLoader(x, y_src, y_ref)

    # Two iterations, checkpoint at the end; no eval (val_loader is None).
    _build_solver(tmp_path).train(
        loader, total_iters=2, log_every=1, save_every=2, eval_every=1000
    )
    ckpts = tmp_path / "checkpoints"
    # The full live nets, EMA nets and optimizers are all saved (needed to resume).
    assert list(ckpts.glob("000002_nets.ckpt"))
    assert list(ckpts.glob("000002_nets_ema.ckpt"))
    assert list(ckpts.glob("000002_optims.ckpt"))

    # A fresh solver resumes from step 2 and runs to step 4.
    _build_solver(tmp_path).train(
        loader, resume_iter=2, total_iters=4, log_every=1, save_every=2, eval_every=1000
    )
    assert list(ckpts.glob("000004_nets.ckpt"))
    assert list(ckpts.glob("000004_optims.ckpt"))
