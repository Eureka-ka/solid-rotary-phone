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


## ✅ 已完成：第一问·压降机理模型（Python 实现）

### 模型概要（参考思路：多段串联）
    dP = dP_manifold + dP_channel + dP_pin + dP_turn
- 沿程：Darcy-Weisbach dP_f = f*(L/D_h)*(rho*V^2/2)，f=64/Re（层流）/Blasius
- 针肋：dP_pin = (M*n*fin_per_row) * C_D * (0.5*rho*V_gap^2) * (d_pin*H)/A_c，C_D=1+10/Re_d^0.8
- 歧管：dP_man = (f_man*L_man/D_man + 1) * 0.5*rho*V_man^2，V_man=mdot/(rho*W_man*H_man)
- 转弯：dP_turn = K_turn * 0.5*rho*V^2

### 关键假设（与热阻模型不同处，需在论文中说明）
- 压降模型假设**微通道层深度 H_ch 固定**（=H_ch_ref=0.5mm），歧管深高比 r 只改变歧管层
  深度 H_man=r*H_ch —— 这是"P 随 r 增大而下降"（歧管越深→歧管压降越小）成立的关键；
- 歧管分配槽宽度 W_man=0.3mm（较窄，使歧管压降占主导），属可标定典型值；
- 数据标定：P*_pred = c0 + c_f*dP_f + c_pin*dP_pin + c_man*dP_man + c_turn*dP_turn（最小二乘）

### 验证结果（附件2 84组样本）
| 指标 | 数值 |
|---|---|
| Pearson（原始 dP_total vs 数据） | 0.8801 |
| Spearman | 0.8425 |
| R²（标定后） | 0.8853 |
| RMSE / 平均相对误差 | 0.00935 / 6.54% |
| 模型 dP_total 范围 | 5696~13735 Pa |
| 标定后分量贡献 | 歧管 52.7%、针肋 26.4%、沿程 20.9%、转弯 0% |

趋势结论（模型与数据一致）：
- P 随 w 增大而上升（针肋形阻 + 间隙流速增大）
- P 随 r 增大而下降（歧管越深，歧管压降越小）—— r 趋势几乎完美
- P 随 n 增大而上升（针肋排数增多）

### 文件
- `code/pressure_drop_model.py` — 压降模型 PressureDropModel / PressureGeometry
- `run_pressure_drop.py` — 主程序：计算→最小二乘标定→验证→出图
- `outputs/pressure_drop_model_predictions.csv`、`outputs/pressure_drop_metrics.txt`
- `outputs/pressure_drop_trends.png` / `_parity.png` / `_decomposition.png`

### 运行方式
    python run_pressure_drop.py


## ✅ 已完成：第一问·温度非均匀性机理模型（Python 实现）

### 模型概要（参考思路：分配均匀性 + 局部强化均匀性）
指标 U*（无量纲温度非均匀性，对应附件2 U）：
    U* = c0 + c1*x_mal + c2*x_w + c3*x_n + c4*n + c5*(r - r_opt)^2
- x_mal = (0.5*rho*V_man^2)/(dP_f + dP_pin) ：分配不均匀数（歧管动压/通道压降）
    · r 越大（歧管越深）→ V_man 越小 → x_mal 越小 → 分配越均匀 → U 越小
    · w、n 越大 → 通道压降越大 → x_mal 越小（针肋混合/均匀化收益）
- x_w=(w-w_opt)^2、x_n=(n-n_opt)^2：局部强化非均匀（w_opt≈0.2、n_opt≈4 取极小）
- c4*n：针肋排数线性驱动的局部扰动/分配不均
- c5*(r-r_opt)^2：歧管最优深度（r_opt≈3.5），过浅分配不均、过深流动扰动

### 验证结果（附件2 84组样本，最小二乘标定）
| 指标 | 数值 |
|---|---|
| R²（标定后） | 0.7594 |
| RMSE / 平均相对误差 | 0.00948 / 0.95% |
| 模型 U* 范围 | 0.784~0.871（数据 0.774~0.873）|

趋势结论（模型与数据一致，均含非单调）：
- U 随 w 先降后升，w≈0.2 最小；随 n 先降后升，n≈4 最小
- U 随 r 先降后微升，r≈3.5~4 最小（歧管最优深度）

### 文件
- `code/nonuniformity_model.py` — NonuniformityModel
- `run_nonuniformity.py` — 主程序：特征→LS标定→验证→出图
- `outputs/nonuniformity_model_predictions.csv`、`nonuniformity_metrics.txt`
- `outputs/nonuniformity_trends.png` / `_parity.png` / `_contrib.png`

### 运行方式
    python run_nonuniformity.py

## ✅ 第一问已完成（三个机理模型 + 总结文档）
- 热阻 R²=0.733 / 压降 R²=0.885 / 温度非均匀性 R²=0.759
- 影响规律与数据互验总表、三项指标合理性论证：见 `docs/Q1_第一问总结.md`

## ⏭️ 下一步（待办）
- Q2：代理模型（RSM/克里金/RF/BP + LOOCV）
- Q3~Q5：优化、权重敏感性、不确定性分析


