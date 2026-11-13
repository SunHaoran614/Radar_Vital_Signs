# Radar Vital Signs

毫米波雷达心率估计：数据读取、可替换的 RDA 前端、时频表征、深度模型训练与无标签域适配。训练和验证在本机进行。

本仓库发布可维护的运行代码。历史实验、一次性脚本、审计、结果、论文草稿和复现快照集中保存在本地 `historical_archive/`，不纳入 Git、Python 安装包或 Docker 构建上下文。保留代码不代表其方法已证明跨域有效。

## 安装

Python 3.10 及以上，建议 Python 3.12。PyTorch 的 GPU 版本按本机驱动安装。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

已有本地 GPU 镜像可用 `docker compose -f compose.local.yml up -d`。通用镜像可用 `docker compose build` 构建；实际数据挂载路径按本机配置。

## 使用

```bash
radar-export --help
radar-build-windows --help
radar-train --help
radar-evaluate --help
radar-adapt --help
radar-infer --help
```

也可从项目根目录用 `python -m src.inference.dar_pipeline --help` 等模块命令运行。为保持调用兼容，Python 包名沿用 `src`。

- [数据格式与数据约束](docs/data.md)
- [RDA 前端与表征](docs/frontend.md)
- [训练、评估和适配](docs/training.md)
- [固定模型推理](docs/inference.md)
- [已审阅论文图片](docs/manuscript/figures/reviewed/README.md)

数据和权重由使用者单独提供，不随源码分发。固定推理需要明确的权重 SHA256；项目不会自动下载数据、读取目标参考或选择权重。

## 目录

```text
src/
  data/         数据读取与窗口构建
  radar/        ADC/RDA 前端算法池
  features/     相位、分解与表征
  models/       共享模型与对比骨干
  training/     训练、评估与适配
  inference/    固定权重推理
  evaluation/   记录/受试者统计
configs/        可复用配置
tests/         无需私有数据的测试
scripts/        发布范围检查工具
docs/           使用说明与已审阅图片
```

本地数据存放在 `data/` 或外部路径。新的窗口和运行输出使用显式参数指定目录；数据、权重、日志及生成物均排除在 Git 之外。

## 开发与发布检查

```bash
python -m pytest -q
python scripts/check_release.py
```

发布检查覆盖目录范围、Python 语法、内部导入、文档链接、归档忽略规则和 Git 跟踪范围；不能替代模型精度评估。归档不会重写已有 Git 提交历史，也不替代独立备份。

数据角色、标签隔离及原始窗口 ID 配对要求见 [AGENTS.md](AGENTS.md)。本项目尚未提供对新设备或临床场景的精度保证。
