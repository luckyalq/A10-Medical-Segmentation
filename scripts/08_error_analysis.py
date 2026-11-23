from __future__ import annotations

import argparse
import csv
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from _shared import PROJECT_ROOT, make_dataset
from utils.common import load_config, project_path, save_json


def safe_score(function, target: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(function(target.reshape(-1), score.reshape(-1)))
    except ValueError:
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="分析不确定性与分割错误、漏检区域的关联")
    parser.add_argument("--method", choices=["ensemble", "mc_dropout"], required=True)
    parser.add_argument("--metric", choices=["entropy", "variance", "mutual_information"], default="entropy")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = make_dataset(config, "test")
    targets = {str(dataset[index]["id"]): dataset[index]["mask"].squeeze().numpy() for index in range(len(dataset))}
    prediction_dir = project_path(config, "results", "predictions", args.method, "test")
    uncertainty_dir = project_path(config, "results", "uncertainty_maps", args.method, "test")
    output_dir = project_path(config, "results", "error_maps", args.method)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    quantile_errors: list[list[float]] = [[] for _ in range(5)]
    threshold = float(config["training"]["threshold"])
    for uncertainty_path in tqdm(sorted(uncertainty_dir.glob("*.npz")), desc="error analysis"):
        sample_id = uncertainty_path.stem
        if sample_id not in targets:
            continue
        probability = np.load(prediction_dir / uncertainty_path.name)["mean_probability"]
        uncertainty = np.load(uncertainty_path)[args.metric]
        truth = targets[sample_id] >= 0.5
        prediction = probability >= threshold
        error = prediction != truth
        false_negative = np.logical_and(truth, ~prediction)
        boundaries = np.quantile(uncertainty.reshape(-1), [0.2, 0.4, 0.6, 0.8])
        groups = np.digitize(uncertainty, boundaries)
        group_rates = []
        for group in range(5):
            selected = groups == group
            rate = float(error[selected].mean()) if selected.any() else math.nan
            quantile_errors[group].append(rate)
            group_rates.append(rate)
        top_threshold = np.quantile(uncertainty, 0.8)
        fn_total = int(false_negative.sum())
        fn_covered = int(np.logical_and(false_negative, uncertainty >= top_threshold).sum())
        rows.append({
            "id": sample_id,
            "error_auroc": safe_score(roc_auc_score, error, uncertainty),
            "error_auprc": safe_score(average_precision_score, error, uncertainty),
            "error_rate": float(error.mean()),
            "fn_pixels": fn_total,
            "fn_top20_coverage": fn_covered / max(fn_total, 1),
            **{f"q{index + 1}_error_rate": group_rates[index] for index in range(5)},
        })
        np.savez_compressed(output_dir / uncertainty_path.name, error=error.astype(np.uint8), false_negative=false_negative.astype(np.uint8), uncertainty=uncertainty.astype(np.float32))
    if not rows:
        raise RuntimeError("没有可分析的测试结果，请先运行 06 和 07")
    with (output_dir / f"{args.metric}_per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": args.method,
        "metric": args.metric,
        "macro_error_auroc": float(np.nanmean([row["error_auroc"] for row in rows])),
        "macro_error_auprc": float(np.nanmean([row["error_auprc"] for row in rows])),
        "macro_fn_top20_coverage": float(np.nanmean([row["fn_top20_coverage"] for row in rows])),
        "uncertainty_quintile_error_rates": [float(np.nanmean(values)) for values in quantile_errors],
    }
    save_json(summary, output_dir / f"{args.metric}_summary.json")
    print(summary)


if __name__ == "__main__":
    main()

