from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from _shared import PROJECT_ROOT
from utils.common import load_config, project_path
from utils.metrics import metrics_from_counts
from utils.visualization import save_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 Coverage-Risk 曲线并计算 AURC")
    parser.add_argument("--method", choices=["ensemble", "mc_dropout"], required=True)
    parser.add_argument("--risk-column", default="entropy_top10_mean")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    risk = pd.read_csv(project_path(config, "results", "uncertainty_maps", args.method, "test", "risk_scores.csv"))
    metrics = pd.read_csv(project_path(config, "results", "metrics", f"{args.method}_test_per_image.csv"))
    data = risk[["id", args.risk_column]].merge(metrics, on="id", validate="one_to_one").sort_values(args.risk_column)
    coverages = np.linspace(0.05, 1.0, 20)
    dice_risk, fnr_risk, rows = [], [], []
    for coverage in coverages:
        count = max(1, int(round(len(data) * coverage)))
        accepted = data.iloc[:count]
        totals = {key: float(accepted[key].sum()) for key in ("tp", "fp", "fn", "tn")}
        aggregate = metrics_from_counts(**totals)
        current_dice_risk = 1.0 - aggregate["dice"]
        dice_risk.append(current_dice_risk)
        fnr_risk.append(aggregate["fnr"])
        rows.append({"coverage": coverage, "review_rate": 1.0 - coverage, "selective_dice_risk": current_dice_risk, "selective_fnr": aggregate["fnr"]})
    output_dir = project_path(config, "results", "metrics")
    with (output_dir / f"{args.method}_risk_coverage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    aurc_dice = float(np.trapz(dice_risk, coverages))
    aurc_fnr = float(np.trapz(fnr_risk, coverages))
    summary = pd.DataFrame([{"method": args.method, "risk_column": args.risk_column, "aurc_dice": aurc_dice, "aurc_fnr": aurc_fnr}])
    summary.to_csv(output_dir / f"{args.method}_aurc.csv", index=False, encoding="utf-8-sig")
    save_curve(coverages, {"1 - Dice": np.asarray(dice_risk), "FNR": np.asarray(fnr_risk)}, "Coverage", "Selective Risk", project_path(config, "results", "figures", f"{args.method}_coverage_risk.png"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

