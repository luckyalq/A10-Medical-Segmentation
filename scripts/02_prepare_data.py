from __future__ import annotations

import argparse
import random

from _shared import PROJECT_ROOT, denormalize
from datasets.isic_dataset import ISICDataset, discover_pairs
from utils.common import data_paths, load_config, project_path, save_json
from utils.seed import set_seed
from utils.visualization import save_segmentation_panel


def write_ids(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成固定 train/val/test 划分")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = args.seed or int(config["project"]["seed"])
    set_seed(seed)
    _, image_dir, mask_dir = data_paths(config)
    pairs, issues = discover_pairs(image_dir, mask_dir)
    if any(issues.values()):
        raise RuntimeError(f"存在未配对数据: {issues}")
    ids = [str(pair["id"]) for pair in pairs]
    random.Random(seed).shuffle(ids)
    total = len(ids)
    train_end = round(total * config["data"]["train_ratio"])
    val_end = train_end + round(total * config["data"]["val_ratio"])
    splits = {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}
    for name, values in splits.items():
        write_ids(project_path(config, "splits", f"{name}.txt"), values)
    save_json({"seed": seed, "total": total, **{name: len(values) for name, values in splits.items()}}, project_path(config, "splits", "split_summary.json"))

    dataset = ISICDataset(image_dir, mask_dir, ids=splits["train"][:1], image_size=config["data"]["image_size"], normalize=config["data"]["normalize"])
    sample = dataset[0]
    save_segmentation_panel(denormalize(sample["image"], config["data"]["normalize"]), sample["mask"].squeeze().numpy(), sample["mask"].squeeze().numpy(), project_path(config, "results", "figures", "data_pair_example.png"))
    print({name: len(values) for name, values in splits.items()})


if __name__ == "__main__":
    main()

