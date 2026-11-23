#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/featurize/work/A10_Medical_Uncertainty"
cd "$PROJECT_ROOT"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/00_environment_check.py
python scripts/01_check_data.py
python scripts/02_prepare_data.py
echo "环境、数据检查和固定划分已经完成。"
echo "下一步: python scripts/03_train_baseline.py"

