# effective_complexity/cli/train.py

import typer
from pathlib import Path

from effective_complexity.utils.io import load_config
from effective_complexity.data.datasets import get_dataloaders
from effective_complexity.experiments.runner import run_multiple_experiments
from effective_complexity.utils.results import average_results_across_seeds

app = typer.Typer()


@app.command()
def run(out: str = "runs"):
    """
    Run full multi-seed multi-architecture training.
    """

    cfg = load_config()

    # -----------------------
    # Dataset
    # -----------------------

    loaders = get_dataloaders(cfg["data"])

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Run experiments
    # -----------------------

    results = run_multiple_experiments(
        cfg=cfg["training"],
        architectures=cfg["architectures"],
        seeds=cfg["seeds"],
        loaders=loaders,
        out_dir=str(out),
    )

    # -----------------------
    # Average across seeds
    # -----------------------

    avg = average_results_across_seeds(
        results,
        save_dir=out / "averaged"
    )

    typer.echo("Training complete.")
    typer.echo(f"Averaged results saved to {out/'averaged'}")
