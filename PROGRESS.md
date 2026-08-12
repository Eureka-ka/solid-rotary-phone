# PROGRESS.md — 项目进度记录（选题B）

本文件记录已完成的建模工作，供后续会话/用户查阅。建模思路与约定见 `AGENTS.md`。

## ✅ 已完成：第一问·热阻机理模型（Python 实现）

### 模型概要
采用参考思路的串联热阻分解：
    R_total = R_cond + R_conv + R_fluid
- R_cond = delta_sub / (k_AlN * A_foot)      （衬底导热，结构无关）
- R_conv = 1 / (h_eff * A_eff)               （对流，w/r/n 作用主战场）
- R_fluid = 1 / (2*mdot*Cp) ≈ 0.1196 K/W     （流体升温热阻，结构无关下限之一）

关键物理要素：
- 通道壁面 Nu：层流 4.36 / 过渡湍流 Gnielinski
- 针肋 Nu：Zukauskas 单圆柱横掠关联式（分段系数）
- 圆柱针肋效率 eta = tanh(mH)/(mH)
- 针肋阻塞/尾流修正：A_fin_eff = A_fin * (1 - cb*(w/w_max)^p)，cb=0.5, p=8（半经验，由数据标定）
- 微通道深度随歧管深高比变化：H_ch(r) = H_tot/(1+r)（总高度近似固定）
- 无量纲热阻（工程常用定义，与数据带一致）：R* = R_total * mdot * Cp

### 默认几何（图A1/A2未给全尺寸，取典型值并参数化）
N_ch=100, w_ch=0.08mm, H_ch_ref=0.5mm, M=4, delta_sub=0.2mm, fin_per_row=3

### 验证结果（附件2 84组样本，仿射标定 R*_pred = c0 + c1*R*）
| 指标 | 数值 |
|---|---|
| Pearson（原始模型 vs 数据） | 0.8564 |
| Spearman | 0.868 |
| R²（标定后） | 0.7333 |
| RMSE | 0.00578 |
| 平均相对误差 | 0.58% |
| 模型 R* 范围 | 0.6915~0.7800 |
| 数据 R 范围 | 0.7219~0.7738 |

趋势结论（模型与数据一致）：
- R 随 w 先降后微升，w≈0.2 处最小（针肋阻塞效应）
- R 随 r 增大而上升（微通道变浅、换热面积减小）
- R 随 n 增大而下降（针肋面积/扰动增强）

### 文件
- `code/thermal_resistance_model.py` — 机理模型类 ThermalResistanceModel / Geometry
- `run_thermal_resistance.py` — 主程序：加载数据→计算→标定→验证→出图
- `outputs/thermal_R_model_predictions.csv` — 84组逐样本预测（含 R_cond/R_conv/R_fluid/R_total/R*）
- `outputs/thermal_R_metrics.txt` — 验证指标与趋势表
- `outputs/thermal_R_trends.png` — 趋势对比图（模型 vs 数据）
- `outputs/thermal_R_parity.png` — 标定前后散点对比
- `outputs/thermal_R_decomposition.png` — 典型方案热阻分解

### 运行方式
    python run_thermal_resistance.py

## ⏭️ 下一步（待办）
- 第一问 B 部分：压降模型、温度非均匀性模型（机理），以及三项指标合理性论证
- 第一问收尾：影响规律与数据互验表（可基于热阻模型结果展开）
- Q2：代理模型（RSM/克里金/RF/BP + LOOCV）
- Q3~Q5：优化、权重敏感性、不确定性分析
