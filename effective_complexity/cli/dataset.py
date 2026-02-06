# effective_complexity/cli/dataset.py

import typer
import torch
from pathlib import Path

from effective_complexity.data.datasets import get_dataloaders
from effective_complexity.utils.io import load_config

app = typer.Typer()


@app.command()
def generate(out: str = "datasets"):
    """
    Generate and save dataset to disk using default config.
    """

    cfg = load_config()
    loaders = get_dataloaders(cfg["data"])

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    for split, loader in loaders.items():

        xs, ys = [], []

        for batch in loader:
            xs.append(batch["x"])
            ys.append(batch["label"])

        xs = torch.cat(xs)
        ys = torch.cat(ys)

        torch.save(
            {"x": xs, "label": ys},
            out / f"{split}.pt"
        )

        typer.echo(f"Saved {split}: {xs.shape}")

    typer.echo("Dataset generation complete.")
