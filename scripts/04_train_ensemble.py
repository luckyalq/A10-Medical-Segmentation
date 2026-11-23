from __future__ import annotations

import argparse
import copy

from _shared import PROJECT_ROOT, build_unet, make_loader
from utils.common import get_device, load_config, project_path
from utils.engine import train_model
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="独立训练 Deep Ensemble 的 U-Net 成员")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--outer-seed", type=int, default=0, help="外层重复实验编号，仅用于隔离输出目录")
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true", help="从各成员last检查点继续训练")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds or list(config["ensemble"]["seeds"])
    device = get_device(args.device)
    folder = f"outer_{args.outer_seed}" if args.outer_seed else "default"
    for seed in seeds:
        effective_seed = int((seed + args.outer_seed * 1009) % 2147483647)
        print(f"Training ensemble member label={seed}, effective_seed={effective_seed}")
        set_seed(effective_seed)
        member_config = copy.deepcopy(config)
        member_config["project"]["seed"] = effective_seed
        model = build_unet(config)
        train_model(model, make_loader(config, "train", augment=True, shuffle=True), make_loader(config, "val"), device, member_config, project_path(config, "checkpoints", "ensemble", folder, f"seed_{seed}.pth"), project_path(config, "logs", "ensemble", folder, f"seed_{seed}.csv"), resume=args.resume)


if __name__ == "__main__":
    main()

