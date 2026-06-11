# 翼型外形优化

[English](README.md) | [中文](README.cn.md)

**基于贝叶斯优化与 RANS CFD 的多目标翼型优化工具 —— 寻找多个攻角下最优升阻比。**

## 功能简介

设计更好的机翼。本工具使用 **CST 参数化**描述翼型外形，借助 **Optuna 的 TPE 采样器**智能探索设计空间，并通过 **OpenFOAM RANS (simpleFoam)** 以 CFD 评估每个候选方案。最终输出在您关注的各攻角下最大化 L/D 的 **Pareto 前沿**翼型。

- 🧬 **CST 翼型参数化** —— 仅需少量系数即可生成光滑、真实的翼型
- 🧠 **贝叶斯优化 (TPE)** —— 相比暴力搜索或遗传算法，用更少的 CFD 运行次数找到优秀设计
- 🌪️ **RANS CFD 评估** —— 二维不可压缩流动，支持 MPI 并行 OpenFOAM
- 📊 **实时 Pareto 前沿** —— 随试验进行实时观察权衡曲面演化
- 🖥️ **PySide6 图形界面** —— 无需命令行即可配置、运行和监控

## 快速开始

```bash
# 1. 安装 uv（仅一次）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库并配置环境
git clone <此仓库地址>
cd airfoil_profile_optimization
uv sync

# 3. 启动
./start.sh
```

> **前置依赖：** Python 3.11+ 以及安装有 MPI 的 [OpenFOAM](https://openfoam.org/)。在配置中将 `solver_env_path` 指向您的 OpenFOAM 安装目录。

在 GUI 中选择工作目录，调整设计参数，然后点击 **Start**。

## 工作原理

```
CST 系数 → Gmsh 网格 → RANS 求解 → Cl, Cd → L/D
       ↑                                         ↓
       └────────── Optuna TPE 建议下一组 ←──────────┘
```

1. **参数化** —— 用 CST Bernstein 系数定义翼型上下表面
2. **网格生成** —— Gmsh 生成翼型周围的结构化远场网格
3. **仿真** —— OpenFOAM simpleFoam 在每个攻角下求解 RANS（各攻角并行计算）
4. **优化** —— Optuna 将 L/D 反馈给 TPE 采样器，指导下一轮建议

## 输出文件

所有文件均保存在您指定的工作目录中：

| 内容 | 路径 |
|------|------|
| 优化数据库 | `logs/airfoil_optim.db` |
| Pareto 前沿 CSV | `logs/pareto_front.csv` |
| 单次试验图 | `logs/figs/trial_*.png` |
| CAD 几何 (STEP) | `sims/airfoil.stp` |
| 提取的试验数据 | `solution/trial_*/` |

## 项目结构

```
src/
├── frontend.py          # PySide6 图形界面
├── backend.py           # Optuna 优化循环
├── config.py            # 配置管理
├── tool_mesher.py       # Gmsh 网格生成
├── tool_simulator.py    # OpenFOAM RANS 案例设置与运行
├── tool_postprocessor.py# forceCoeffs → Cl, Cd, L/D
├── tool_cader.py        # CadQuery STEP 导出
└── visualizer.py        # 试验图与 Pareto 可视化
sims/sim_ref/            # OpenFOAM 模板案例
```

## 许可证

MIT
