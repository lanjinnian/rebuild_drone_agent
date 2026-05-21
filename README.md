# Rebuild Drone Agent

无人机视频、图像与遥感约束驱动的三维重建实验项目。当前仓库已按模块化流程搭建基础结构，后续可逐步接入 VGGT/VGGT-Long、图像匹配、分块对齐、融合导出与语义理解能力。

## Project Layout

- `configs/`: 默认参数、数据集、重建和优化配置。
- `data/`: 原始数据、中间结果、关键帧、分块序列和临时输出。
- `src/pipeline/`: 总流程调度。
- `src/preprocessing/`: 抽帧、关键帧选择、模糊过滤和动态目标过滤。
- `src/reconstruction/`: VGGT/VGGT-Long 重建、深度转点云和置信度过滤。
- `src/matching/`: 局部匹配、遥感图匹配和闭环检测。
- `src/optimization/`: Sim3、Umeyama、全局 LM 和位姿图优化。
- `src/fusion/`: 点云、网格和 3DGS 导出。
- `src/semantic/`: VLM 解析、目标检测和场景资产管理。
- `scripts/`: 常用命令行入口。
- `app/`: Gradio UI 与 API 服务占位。
- `experiments/`: 实验记录与输出目录。
- `results/`: 图表、指标、点云和演示结果。
- `tests/`: 单元测试入口。
- `docs/`: 系统设计、算法笔记、实验日志和使用文档。

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config configs/default.yaml
```

## Current Status

本次提交主要完成项目结构搭建和基础占位实现。各算法模块提供稳定的函数/类入口，便于后续逐步替换为真实实现。
