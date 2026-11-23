from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from _shared import PROJECT_ROOT, make_dataset
from utils.common import load_config, project_path
from utils.metrics import metrics_from_counts, numpy_counts


def sum_counts(records: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(sum(record[key] for record in records)) for key in ("tp", "fp", "fn", "tn")}


def main() -> None:
    parser = argparse.ArgumentParser(description="使用验证集阈值模拟 Abstain 与人工复核")
    parser.add_argument("--method", choices=["ensemble", "mc_dropout"], required=True)
    parser.add_argument("--risk-column", default="entropy_top10_mean")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    val_path = project_path(config, "results", "uncertainty_maps", args.method, "val", "risk_scores.csv")
    test_path = project_path(config, "results", "uncertainty_maps", args.method, "test", "risk_scores.csv")
    validation = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    if args.risk_column not in validation or args.risk_column not in test:
        raise KeyError(f"风险列不存在: {args.risk_column}")
    dataset = make_dataset(config, "test")
    targets = {str(dataset[index]["id"]): dataset[index]["mask"].squeeze().numpy() for index in range(len(dataset))}
    predictions = project_path(config, "results", "predictions", args.method, "test")
    threshold = float(config["training"]["threshold"])
    original: dict[str, dict[str, float]] = {}
    lesion_pixels: dict[str, float] = {}
    background_pixels: dict[str, float] = {}
    for sample_id, truth in targets.items():
        probability = np.load(predictions / f"{sample_id}.npz")["mean_probability"]
        original[sample_id] = numpy_counts(probability, truth, threshold)
        lesion_pixels[sample_id] = float((truth >= 0.5).sum())
        background_pixels[sample_id] = float((truth < 0.5).sum())
    baseline_counts = sum_counts(list(original.values()))
    baseline_metrics = metrics_from_counts(**baseline_counts)
    output_dir = project_path(config, "results", "abstain", args.method)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    decision_rows = []
    for review_rate in config["uncertainty"]["review_rates"]:
        # 阈值只由验证集确定，测试标签不参与工作点选择。
        risk_threshold = float(validation[args.risk_column].quantile(1.0 - float(review_rate)))
        rejected = set(test.loc[test[args.risk_column] > risk_threshold, "id"].astype(str))
        accepted = set(test["id"].astype(str)) - rejected
        accepted_counts = sum_counts([original[sample_id] for sample_id in accepted])
        rejected_original_counts = sum_counts([original[sample_id] for sample_id in rejected])
        actual_review_rate = len(rejected) / max(len(test), 1)
        for sample_id in test["id"].astype(str):
            decision_rows.append({"review_budget": review_rate, "id": sample_id, "risk_score": float(test.loc[test["id"].astype(str) == sample_id, args.risk_column].iloc[0]), "risk_threshold": risk_threshold, "decision": "abstain" if sample_id in rejected else "accept"})
        scenarios = {"oracle": 1.0, "correct_90pct_fn": 0.90, "correct_80pct_fn": 0.80}
        for scenario, correction_rate in scenarios.items():
            if scenario == "oracle":
                # Ground Truth仅模拟理想人工复核上限。
                reviewed = {
                    "tp": sum(lesion_pixels[sample_id] for sample_id in rejected),
                    "fp": 0.0,
                    "fn": 0.0,
                    "tn": sum(background_pixels[sample_id] for sample_id in rejected),
                }
            else:
                corrected_fn = rejected_original_counts["fn"] * correction_rate
                reviewed = dict(rejected_original_counts)
                reviewed["fn"] -= corrected_fn
                reviewed["tp"] += corrected_fn
            system_counts = {key: accepted_counts[key] + reviewed[key] for key in accepted_counts}
            system_metrics = metrics_from_counts(**system_counts)
            reduction = (baseline_metrics["fnr"] - system_metrics["fnr"]) / max(baseline_metrics["fnr"], 1e-7)
            summary_rows.append({
                "method": args.method, "risk_column": args.risk_column, "review_budget": review_rate,
                "actual_review_rate": actual_review_rate, "risk_threshold_from_val": risk_threshold,
                "scenario": scenario, "baseline_fnr": baseline_metrics["fnr"],
                "accepted_subset_fnr": metrics_from_counts(**accepted_counts)["fnr"],
                "rejected_subset_original_fnr": metrics_from_counts(**rejected_original_counts)["fnr"],
                "reviewed_system_fnr": system_metrics["fnr"], "fnr_relative_reduction": reduction,
                "reviewed_system_dice": system_metrics["dice"], "reviewed_system_iou": system_metrics["iou"],
            })
    with (output_dir / "threshold_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)
    with (output_dir / "image_decisions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0].keys()))
        writer.writeheader(); writer.writerows(decision_rows)
    pd.DataFrame(decision_rows).query("decision == 'accept'").to_csv(output_dir / "accepted_images.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(decision_rows).query("decision == 'abstain'").to_csv(output_dir / "rejected_images.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / "image_risk_scores.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()

