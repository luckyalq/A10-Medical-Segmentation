from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取YAML并注入项目根目录，避免依赖当前命令行所在位置。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path.resolve())
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_project_relative(config: dict[str, Any], value: str | Path) -> Path:
    """展开环境变量，并把相对路径解释为相对于项目根目录。"""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded.resolve()
    return (Path(config["_project_root"]) / expanded).resolve()


def data_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """返回数据根目录、图像目录和Mask目录。

    环境变量A10_DATA_ROOT优先于config.yaml，便于适配服务器数据挂载点。
    """
    if os.environ.get("A10_DATA_ROOT"):
        root = Path(os.environ["A10_DATA_ROOT"]).expanduser().resolve()
    else:
        root = resolve_project_relative(config, config["data"]["root"])
    return root, root / config["data"]["image_subdir"], root / config["data"]["mask_subdir"]


def project_path(config: dict[str, Any], key: str, *parts: str) -> Path:
    base = resolve_project_relative(config, config["paths"][key])
    return base.joinpath(*parts)


def ensure_directories(config: dict[str, Any]) -> None:
    for key in ("splits", "checkpoints", "results", "logs"):
        project_path(config, key).mkdir(parents=True, exist_ok=True)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def save_checkpoint(path: str | Path, model: torch.nn.Module, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **metadata}, path)

