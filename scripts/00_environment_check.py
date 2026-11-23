"""训练前检查项目路径、数据目录、CUDA环境和模型输出尺寸。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.vgg_unet import build_segmentation_model
from utils.common import data_paths, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="检查Featurize环境、数据路径和模型结构")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root, image_dir, mask_dir = data_paths(config)
    print(f"项目根目录 : {PROJECT_ROOT}")
    print(f"数据根目录 : {data_root}")
    print(f"图像目录   : {image_dir}  存在={image_dir.is_dir()}")
    print(f"Mask目录   : {mask_dir}  存在={mask_dir.is_dir()}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA可用   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本   : {torch.version.cuda}")
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit("数据目录不存在；若数据不在默认位置，请设置 A10_DATA_ROOT。")
    if args.skip_forward:
        return
    cfg = config["model"]
    model = build_segmentation_model(cfg["architecture"], cfg["in_channels"], cfg["out_channels"], cfg["base_channels"], cfg["dropout"], pretrained=False).eval()
    with torch.inference_mode():
        size = config["data"]["image_size"]
        output = model(torch.zeros(1, 3, size, size))
    expected = (1, 1, size, size)
    if tuple(output.shape) != expected:
        raise RuntimeError(f"模型输出尺寸错误: {tuple(output.shape)}，期望{expected}")
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"模型结构   : {cfg['architecture']}")
    print(f"参数量     : {count / 1e6:.2f} M")
    print(f"输出尺寸   : {tuple(output.shape)}")
    print("环境自检通过。")


if __name__ == "__main__":
    main()

