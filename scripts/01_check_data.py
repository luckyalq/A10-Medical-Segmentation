from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from _shared import PROJECT_ROOT
from datasets.isic_dataset import discover_pairs
from utils.common import data_paths, load_config, project_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 ISIC2018 图像与 Mask 的完整性和配对关系")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_root, image_dir, mask_dir = data_paths(config)
    print(f"数据根目录: {data_root}")
    pairs, issues = discover_pairs(image_dir, mask_dir)
    output_dir = project_path(config, "results", "metrics", "data_check")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    invalid = 0
    for pair in tqdm(pairs, desc="checking"):
        record = {"id": pair["id"], "image": str(pair["image"]), "mask": str(pair["mask"]), "status": "ok"}
        try:
            with Image.open(pair["image"]) as source:
                source.verify()
            with Image.open(pair["image"]) as source:
                image_size = source.size
            with Image.open(pair["mask"]) as source:
                mask = np.asarray(source.convert("L"))
                mask_size = source.size
            unique_values = np.unique(mask)
            record.update(
                image_width=image_size[0], image_height=image_size[1], mask_width=mask_size[0], mask_height=mask_size[1],
                same_size=image_size == mask_size, mask_min=int(mask.min()), mask_max=int(mask.max()),
                mask_unique_count=int(unique_values.size), foreground_fraction=float((mask >= 128).mean()),
            )
            if image_size != mask_size or unique_values.size < 2:
                record["status"] = "warning"
        except Exception as exc:
            record["status"] = "invalid"
            record["error"] = str(exc)
            invalid += 1
        records.append(record)
    fields = sorted({key for row in records for key in row})
    with (output_dir / "pair_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {"paired": len(pairs), "invalid": invalid, **issues}
    save_json(summary, output_dir / "data_check_report.json")
    print(summary)
    if invalid or issues["images_without_masks"] or issues["masks_without_images"]:
        raise SystemExit("数据检查发现阻塞问题，请查看检查报告")


if __name__ == "__main__":
    main()

