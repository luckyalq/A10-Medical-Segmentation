from __future__ import annotations

import argparse
import csv

import numpy as np
from tqdm import tqdm

from _shared import PROJECT_ROOT, denormalize, make_dataset
from utils.common import load_config, project_path
from utils.uncertainty import image_risk_scores, uncertainty_maps
from utils.visualization import save_uncertainty_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="计算预测熵、方差、互信息与图像级风险分数")
    parser.add_argument("--method", choices=["baseline", "ensemble", "mc_dropout"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    prediction_dir = project_path(config, "results", "predictions", args.method, args.split)
    if not prediction_dir.exists():
        raise FileNotFoundError(f"请先生成预测: {prediction_dir}")
    output_dir = project_path(config, "results", "uncertainty_maps", args.method, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(config, args.split)
    by_id = {str(dataset[index]["id"]): dataset[index] for index in range(len(dataset))}
    rows = []
    for path in tqdm(sorted(prediction_dir.glob("*.npz")), desc="uncertainty"):
        sample_id = path.stem
        data = np.load(path)
        # samples维度为[T,H,W]，T对应5个成员或20次MC前向。
        maps = uncertainty_maps(data["samples"])
        np.savez_compressed(output_dir / path.name, **maps)
        row = {"id": sample_id}
        for metric in ("entropy", "variance", "mutual_information"):
            # 三种聚合方式会在消融实验中比较。
            for name, value in image_risk_scores(maps[metric], float(config["uncertainty"]["high_quantile"])).items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
        if len(rows) <= 8 and sample_id in by_id:
            item = by_id[sample_id]
            save_uncertainty_panel(denormalize(item["image"], config["data"]["normalize"]), item["mask"].squeeze().numpy(), maps["mean_probability"], maps, project_path(config, "results", "figures", f"{args.method}_{args.split}_uncertainty", f"{sample_id}.png"))
    if not rows:
        raise RuntimeError(f"未找到预测文件: {prediction_dir}")
    with (output_dir / "risk_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} uncertainty records to {output_dir}")


if __name__ == "__main__":
    main()

