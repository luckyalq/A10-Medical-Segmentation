from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _shared import PROJECT_ROOT, denormalize, load_model, make_loader
from models.ensemble import EnsemblePredictor
from models.mc_dropout import MCDropoutPredictor
from utils.common import get_device, load_config, project_path, save_json
from utils.calibration import StreamingECE
from utils.metrics import Confusion, numpy_counts
from utils.visualization import save_reliability_diagram, save_segmentation_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="生成基线、Ensemble 或 MC-Dropout 概率预测")
    parser.add_argument("--method", choices=["baseline", "ensemble", "mc_dropout"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ensemble-folder", default="default")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = args.seed or int(config["project"]["seed"])
    device = get_device(args.device)
    loader = make_loader(config, args.split)
    checkpoint_root = project_path(config, "checkpoints")

    if args.method == "baseline":
        model = load_model(config, checkpoint_root / "baseline" / f"seed_{seed}.pth", device)
        predictor = None
    elif args.method == "ensemble":
        members = [load_model(config, checkpoint_root / "ensemble" / args.ensemble_folder / f"seed_{member_seed}.pth", device) for member_seed in config["ensemble"]["seeds"]]
        predictor = EnsemblePredictor(members).to(device)
        model = None
    else:
        model = load_model(config, checkpoint_root / "mc_dropout" / f"seed_{seed}.pth", device, dropout=float(config["model"]["dropout"]))
        predictor = MCDropoutPredictor(model, int(config["mc_dropout"]["passes"]))

    output_dir = project_path(config, "results", "predictions", args.method, args.split)
    figure_dir = project_path(config, "results", "figures", f"{args.method}_{args.split}_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    confusion = Confusion()
    calibration = StreamingECE(int(config["uncertainty"]["ece_bins"]))
    example_count = 0
    threshold = float(config["training"]["threshold"])
    for batch in tqdm(loader, desc=f"predict {args.method}/{args.split}"):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].cpu().numpy()[:, 0]
        if args.method == "baseline":
            model.eval()
            with torch.inference_mode():
                sample_tensor = torch.sigmoid(model(images)).unsqueeze(0)
        else:
            sample_tensor = predictor.predict_samples(images)
        samples = sample_tensor.detach().cpu().numpy()[:, :, 0]
        means = samples.mean(axis=0)
        confusion.update(torch.from_numpy(means[:, None]), torch.from_numpy(targets[:, None]), threshold)
        calibration.update(means, targets)
        for index, sample_id in enumerate(batch["id"]):
            np.savez_compressed(output_dir / f"{sample_id}.npz", samples=samples[:, index].astype(np.float32), mean_probability=means[index].astype(np.float32))
            metrics = numpy_counts(means[index], targets[index], threshold)
            records.append({"id": sample_id, **metrics})
            if example_count < 8:
                save_segmentation_panel(denormalize(batch["image"][index], config["data"]["normalize"]), targets[index], means[index], figure_dir / f"{sample_id}.png")
                example_count += 1
    metrics_path = project_path(config, "results", "metrics", f"{args.method}_{args.split}_per_image.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    ece, calibration_details = calibration.compute()
    summary = {**confusion.compute(), "ece": ece}
    save_json(summary, project_path(config, "results", "metrics", f"{args.method}_{args.split}_micro.json"))
    save_reliability_diagram(calibration_details, project_path(config, "results", "figures", f"{args.method}_{args.split}_reliability_diagram.png"))
    print(summary)


if __name__ == "__main__":
    main()

