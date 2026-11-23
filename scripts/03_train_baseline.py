from __future__ import annotations

import argparse

from _shared import PROJECT_ROOT, build_unet, make_loader
from utils.common import get_device, load_config, project_path
from utils.engine import train_model
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 U-Net 基线")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true", help="从last检查点继续训练")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = args.seed or int(config["project"]["seed"])
    set_seed(seed)
    device = get_device(args.device)
    model = build_unet(config)
    metrics = train_model(model, make_loader(config, "train", augment=True, shuffle=True), make_loader(config, "val"), device, config, project_path(config, "checkpoints", "baseline", f"seed_{seed}.pth"), project_path(config, "logs", "baseline", f"seed_{seed}.csv"), resume=args.resume)
    print("best validation metrics:", metrics)


if __name__ == "__main__":
    main()

