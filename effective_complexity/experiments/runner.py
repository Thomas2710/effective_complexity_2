# effective_complexity/experiments/runner.py

import os
import json
from effective_complexity.experiments.single_run import run_single_experiment


def run_multiple_experiments(
    base_model_cfg,
    architectures,
    seeds,
    loaders,
    out_dir
):
    """
    base_model_cfg : cfg["model"] from yaml
    architectures  : cfg["architectures"]
    seeds          : list
    """

    results = []

    for arch_name, arch_params in architectures.items():
        for seed in seeds:

            # ---------------------------------
            # Build per-run config
            # ---------------------------------

            run_cfg = base_model_cfg.copy()

            # inject architecture hyperparams
            run_cfg.update(arch_params)

            run_cfg["architecture"] = arch_name
            run_cfg["seed"] = seed

            run_path = os.path.join(
                out_dir,
                arch_name,
                f"seed_{seed}"
            )

            result = run_single_experiment(
                run_cfg,
                loaders,
                run_path
            )

            results.append(result)

    # ---------------------------------
    # Save all results
    # ---------------------------------

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
