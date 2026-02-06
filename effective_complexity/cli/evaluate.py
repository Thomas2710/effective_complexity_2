# effective_complexity/cli/evaluate.py

import typer
import torch
from pathlib import Path

from effective_complexity.models import build_model
from effective_complexity.utils.io import load_config
from effective_complexity.evaluation import (
    run_cca,
    check_diversity,
)

app = typer.Typer()


@app.command()
def model(
    arch: str,
    seed: int = 0,
    runs_dir: str = "runs",
):
    """
    Load a trained model and run quick evaluation.
    """

    cfg = load_config()

    # build model config
    model_cfg = cfg["training"].copy()
    model_cfg.update(cfg["architectures"][arch])

    model = build_model(model_cfg)

    ckpt = Path(runs_dir) / arch / f"seed_{seed}" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    typer.echo(f"Loaded checkpoint: {ckpt}")

    # optional sanity check prints
    typer.echo("Model ready for evaluation.")

