# effective_complexity/utils/results.py

import os
import json
import numpy as np
from collections import defaultdict


def average_results_across_seeds(results, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    by_arch = defaultdict(list)
    for r in results:
        by_arch[r["architecture"]].append(r)

    averaged = {}

    for arch, runs in by_arch.items():

        metrics = defaultdict(list)

        for r in runs:
            for k, v in r.items():
                if k not in ["architecture", "seed"]:
                    metrics[k].append(v)

        avg = {"architecture": arch}

        for k, v in metrics.items():
            arr = np.array(v)
            avg[f"{k}_mean"] = arr.mean(axis=0).tolist()
            avg[f"{k}_std"] = arr.std(axis=0).tolist()

        averaged[arch] = avg

        with open(os.path.join(save_dir, f"{arch}.json"), "w") as f:
            json.dump(avg, f, indent=2)

    return averaged
