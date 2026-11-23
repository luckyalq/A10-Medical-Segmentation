# A10 医学图像分割不确定性量化与失检风险控制

本项目以ISIC2018 Task 1为数据来源，默认使用ImageNet预训练VGG16-BN编码器的U-Net，并比较Deep Ensemble与MC-Dropout的不确定性估计、错误识别和选择性预测表现。经典U-Net保留为消融基线。

服务器部署请先阅读 [Featurize运行说明](docs/FEATURIZE.md)。代码会自动识别项目位置，不包含Windows绝对路径。

## 目录说明

- `configs/`：统一实验配置。
- `splits/`：固定的训练、验证和测试 ID。
- `datasets/`：数据配对、预处理和 DataLoader。
- `models/`：U-Net、Deep Ensemble 和 MC-Dropout。
- `utils/`：训练、指标、校准、不确定性和可视化工具。
- `scripts/`：按编号执行的完整实验流程。
- `checkpoints/`：模型权重。
- `results/`：预测、热力图、统计结果和图片。
- `logs/`：训练日志。

默认从项目同级的 `../data` 读取原始数据，即 `/home/featurize/work/data`。其他挂载位置可用 `A10_DATA_ROOT` 覆盖。所有脚本只读原始数据。

当前磁盘中实际可配对的图像和 Mask 数量由 `01_check_data.py` 动态核验；不要把任务书中的参考数量硬编码进实验。

## 安装

建议使用 Python 3.10 或 3.11，并根据显卡安装匹配的 PyTorch 版本，然后执行：

```powershell
pip install -r requirements.txt
```

## 推荐执行顺序

所有命令均在本项目根目录执行。

```powershell
python scripts/00_environment_check.py
python scripts/01_check_data.py
python scripts/02_prepare_data.py
python scripts/03_train_baseline.py
python scripts/04_train_ensemble.py
python scripts/05_train_mc_dropout.py

python scripts/06_predict.py --method baseline --split test
python scripts/06_predict.py --method ensemble --split val
python scripts/06_predict.py --method ensemble --split test
python scripts/06_predict.py --method mc_dropout --split val
python scripts/06_predict.py --method mc_dropout --split test

python scripts/07_uncertainty_analysis.py --method ensemble --split val
python scripts/07_uncertainty_analysis.py --method ensemble --split test
python scripts/07_uncertainty_analysis.py --method mc_dropout --split val
python scripts/07_uncertainty_analysis.py --method mc_dropout --split test

python scripts/08_error_analysis.py --method ensemble
python scripts/08_error_analysis.py --method mc_dropout
python scripts/09_abstain_simulation.py --method ensemble
python scripts/09_abstain_simulation.py --method mc_dropout
python scripts/10_risk_coverage.py --method ensemble
python scripts/10_risk_coverage.py --method mc_dropout
python scripts/11_ablation.py
```

如显存不足，可在 `configs/config.yaml` 中降低 `training.batch_size`。正式结果需要至少三个外层随机种子；可用各训练脚本的 `--seed` 或 `--outer-seed` 参数分别运行，并保留每次输出。

训练会同时保存最优检查点和last检查点；服务器任务中断后可添加 `--resume` 继续。

例如，Ensemble 的三个外层重复可分别使用 `--outer-seed 42`、`--outer-seed 123`、`--outer-seed 2024`；脚本会在每个外层实验中派生不同的成员初始化，同时将结果写入相互隔离的目录。

## 实验约束

- 图像与 Mask 必须通过 ISIC ID 配对。
- Mask 缩放只使用最近邻插值。
- 训练、验证、测试划分固定为 70%/15%/15%。
- Deep Ensemble 的5个成员必须独立初始化和训练。
- MC-Dropout 推理仅启用 Dropout，BatchNorm 保持评估状态。
- Abstain 阈值只从验证集确定，不能使用测试集标签选阈值。
- 报告指标时同时保留 Micro 汇总与逐图像 Macro 统计。

