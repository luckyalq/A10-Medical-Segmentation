from __future__ import annotations

import argparse
import json

import pandas as pd

from _shared import PROJECT_ROOT
from utils.common import load_config, project_path


def rank_correlation(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank(method="average").corr(b.rank(method="average")))


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总不确定性指标、风险评分和复核预算消融")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    methods = ["ensemble", "mc_dropout"]
    rows = []
    for method in methods:
        risk_path = project_path(config, "results", "uncertainty_maps", method, "test", "risk_scores.csv")
        metric_path = project_path(config, "results", "metrics", f"{method}_test_per_image.csv")
        if not risk_path.exists() or not metric_path.exists():
            print(f"skip {method}: missing predictions or uncertainty results")
            continue
        risk = pd.read_csv(risk_path)
        metrics = pd.read_csv(metric_path)
        data = risk.merge(metrics[["id", "dice", "fnr"]], on="id", validate="one_to_one")
        data["dice_risk"] = 1.0 - data["dice"]
        risk_columns = [column for column in risk.columns if column != "id"]
        for column in risk_columns:
            rows.append({
                "ablation_type": "risk_score", "method": method, "setting": column,
                "spearman_vs_dice_risk": rank_correlation(data[column], data["dice_risk"]),
                "spearman_vs_fnr": rank_correlation(data[column], data["fnr"]),
            })
        abstain_path = project_path(config, "results", "abstain", method, "threshold_results.csv")
        if abstain_path.exists():
            abstain = pd.read_csv(abstain_path)
            for _, record in abstain.iterrows():
                rows.append({
                    "ablation_type": "review_budget", "method": method,
                    "setting": f"{record['scenario']}_budget_{record['review_budget']}",
                    "actual_review_rate": record["actual_review_rate"],
                    "reviewed_system_fnr": record["reviewed_system_fnr"],
                    "fnr_relative_reduction": record["fnr_relative_reduction"],
                })
    if not rows:
        raise RuntimeError("没有可汇总的结果，请先完成 06 至 10")
    output = pd.DataFrame(rows)
    output_path = project_path(config, "results", "metrics", "ablation_results.csv")
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary_parts = []
    for method in methods:
        micro_path = project_path(config, "results", "metrics", f"{method}_test_micro.json")
        aurc_path = project_path(config, "results", "metrics", f"{method}_aurc.csv")
        if micro_path.exists() and aurc_path.exists():
            micro = json.loads(micro_path.read_text(encoding="utf-8"))
            aurc = pd.read_csv(aurc_path).iloc[0].to_dict()
            summary_parts.append({"method": method, **micro, **aurc})
    if summary_parts:
        pd.DataFrame(summary_parts).to_csv(project_path(config, "results", "metrics", "final_summary.csv"), index=False, encoding="utf-8-sig")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

