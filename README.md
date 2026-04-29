# 🚗 电动汽车充电负荷分析与优化系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 基于真实驾驶数据的电动汽车充电行为分析、电价分时策略与充电负荷优化系统

## 📖 项目概述

本项目以电动汽车（EV）充电负荷为核心研究对象，覆盖从 **电价数据分析 → 出行行为建模 → 优化算法实现 → 负荷调度优化 → 可视化** 的完整技术链条。数据源自真实车辆出行记录（17805辆车），包含北京、夏威夷、奥斯陆三地分时电价数据。

---

## 📦 模块说明

项目按功能分为 5 个独立模块（`task1` ~ `task5`），可独立运行，也可串联使用。

### Task 1：🎵 音调播放与音频处理
- **`player.py`** — 8音调循环播放器（基于 Windows `winsound`，C4~C5 全音阶）
- **`playmp3.py`** — MP3 音频播放器功能（基于 `pygame`，自动加载/播放/错误处理）
- **`first.ipynb`** — Jupyter Notebook 交互版本

### Task 2：💡 分时电价数据清洗与可视化
- **`electric.py`** — 电价数据清洗引擎，自动处理：
  - 多货币单位智能换算（CNY/美分/挪威克朗）
  - 区间电价取中值（如 `140-260 øre` → 统一折算）
  - 特殊字符/非标城市名清洗（如「夏威夷 (Oahu)」）
  - 自动安装缺失依赖，开箱即用
- **`visual.py`** — 分时电价可视化系统，生成 3 张图表：
  - `1_分时电价时序折线图.png` — 三城市 24h 电价趋势对比
  - `2_三城市峰谷平电价对比图.png` — 峰谷平三段柱状对比
  - `3_单城市峰谷平电价占比图.png` — 各城市时段电价构成分析
- **`城市电价_清洗完成.csv`** — 清洗后标准结构化数据集

**关键输出**：标准化电价时序数据，支撑后续充电成本优化

### Task 3：📊 电动汽车出行行为统计分析
- **`anay.py`** — 基于 17805 辆车的真实 `DrivingData.xlsx`（Activity 表）进行：
  - 分车聚合日行驶里程、行驶/充电/停车时长、出行频次
  - 描述性统计表（均值、标准差、中位数等）
  - 24h 三状态（出行/充电/停车）概率分布曲线（0.5h 粒度）
  - 停车时长分布 & 充电时长分布
- **输出数据集**：
  - `电动汽车出行行为描述性统计.csv`
  - `电动汽车 24 小时分时状态概率分布.csv`
  - `电动汽车停车时长分布.csv` / `电动汽车充电时长分布.csv`

### Task 4：🧮 二次函数极值求解与优化算法教学
- **`quadratic_maximum.py`** — 以 `y = -x² + 4x + 5` 为例，完整演示：
  - **解析法**：求导 → 驻点 → 二阶导验证 → 精确极值
  - **数值法**：`scipy.optimize.minimize` / `minimize_scalar` / `fminbound` 三种实现
  - **6 张教学可视化图表**：函数曲线、一阶/二阶导数、极值标注、数值优化过程、方法对比表
- **`qmath.ipynb`** — Jupyter Notebook 交互版

**输出**：6 张独立高分辨率图表（`1_~6_*.png`）

### Task 5：⚡ 20辆电动车充电负荷优化调度
- **`eta.py`** — 完整充电负荷分析流水线，功能包括：
  - 读取 `DrivingData_20EVs.xlsx`（20辆车双工作表）
  - **原始负荷计算**与总负荷曲线可视化
  - **波动性分析**：滚动方差法量化负荷波动
  - **优化建模**：基于 `scipy.optimize.minimize` 的二次规划，目标为最小化负荷峰谷差 + 方差
  - **约束条件**：充电需求满足、功率上限、停车时间窗
  - **结果对比**：优化前后负荷曲线叠加、波动性对比、调度计划导出
- **核心输出**：
  - `原始充电负荷.png` / `充电负荷优化效果对比.png`
  - `充电负荷波动性分析.png` + `波动性对比数据.csv`
  - `优化充电调度计划.csv`（每辆车每 15min 功率分配）

---

## 🚀 快速开始

### 环境要求
```bash
Python 3.8+
pip install -r requirements.txt
```

### 运行各模块
```bash
# Task 2：电价数据清洗
python task2/electric.py

# Task 2：电价可视化
python task2/visual.py

# Task 3：出行行为分析
python task3/anay.py

# Task 4：二次函数教学
python task4/quadratic_maximum.py

# Task 5：充电负荷优化
python task5/eta.py -i DrivingData_20EVs.xlsx
```

---

## 📁 项目结构

```
├── README.md
├── requirements.txt
├── 城市电价.xlsx                 # 原始电价数据（北京/夏威夷/奥斯陆）
│
├── task1/                        # 音频播放工具
│   ├── player.py                 # 8音调播放器
│   ├── playmp3.py                # MP3播放器
│   ├── lovesong.mp3              # 示例音频
│   └── first.ipynb               # Notebook版
│
├── task2/                        # 电价数据处理
│   ├── electric.py               # 数据清洗引擎
│   ├── visual.py                 # 可视化系统
│   ├── 城市电价_清洗完成.csv     # 清洗结果数据
│   └── EV_Charts_Optimized/      # 优化图表（24h概率等）
│
├── task3/                        # 出行行为分析
│   ├── anay.py                   # 统计分析核心
│   ├── DrivingData.xlsx          # 17805辆车真实数据
│   ├── 24小时状态概率图.png
│   └── *.csv                     # 统计数据输出
│
├── task4/                        # 优化算法教学
│   ├── quadratic_maximum.py      # 二次函数极值求解
│   ├── qmath.ipynb               # Notebook版
│   └── 1_~6_*.png                # 6张教学图表
│
└── task5/                        # 充电负荷优化
    ├── eta.py                    # 优化调度引擎
    ├── 原始充电负荷基础.csv
    ├── 优化充电调度计划.csv
    ├── 充电负荷优化效果对比.png
    └── ...                       # 其他输出文件
```

---

## 📊 关键技术点

| 模块 | 技术栈 | 亮点 |
|------|--------|------|
| Task2 | `pandas` `matplotlib` | 多货币智能换算、非标数据自动清洗 |
| Task3 | `pandas` `numpy` `openpyxl` | 17805辆车全量统计、0.5h粒度概率分布 |
| Task4 | `scipy.optimize` `matplotlib` | 解析+数值双解法、6图教学体系 |
| Task5 | `scipy.optimize` `pandas` `matplotlib` | 20辆车15min分辨率、二次规划约束优化 |

---

## 📄 License

MIT License
