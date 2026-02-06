from copy import deepcopy
from datetime import datetime

from effective_complexity.data.datasets import get_dataloaders
from effective_complexity.experiments.runner import run_multiple_experiments
from effective_complexity.utils.io import load_config


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = [
    {
        "tag": "C4_T10_S20",
        "num_classes": 4,
        "teacher_d": 10,
        "student_d": 20,
    },
    {
        "tag": "C3_T3_S30",
        "num_classes": 3,
        "teacher_d": 3,
        "student_d": 30,
    },
]


# ============================================================
# MAIN
# ============================================================

def main():

    base_cfg = load_config()

    for exp in EXPERIMENTS:

        cfg = deepcopy(base_cfg)

        # -------- DATA / TEACHER --------
        cfg["data"]["d"] = exp["teacher_d"]
        cfg["data"]["input_size"] = exp["teacher_d"] 
        cfg["data"]["num_classes"] = exp["num_classes"]


        # -------- STUDENT MODEL --------
        cfg["model"]["input_size"] = exp["teacher_d"]
        cfg["model"]["num_classes"] = exp["num_classes"]
        cfg["model"]["embedding_size"] = exp["student_d"]

        loaders = get_dataloaders(cfg["data"])

        out_dir = (
            f"experiments_out/"
            f"{exp['tag']}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        results = run_multiple_experiments(
            base_model_cfg = cfg["model"],
            architectures = cfg["architectures"],
            seeds = cfg["seeds"],
            loaders = loaders,
            out_dir = out_dir,
        )


        print(f"\nFinished experiment: {exp['tag']}")
        print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
