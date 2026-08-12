# -*- coding: utf-8 -*-
"""
温度非均匀性模型 nonuniformity_model.py
========================================
选题B 第一问：温度非均匀性机理模型（参考思路 1.1.doc）。

指标定义：无量纲温度非均匀性 U*（对应附件2 的 U，数据范围 0.774~0.873），
         可用温度标准差 σ_T 或最大温差 (T_max-T_min) 归一化后表示。

两大主导因素（机理）：
1) 流量分配均匀性（歧管深高比 r 主导）
   - 并联通道流量分配的非均匀程度由"歧管动压 / 通道压降"之比（分配不均匀数）决定：
        x_mal = (0.5*rho*V_man^2) / (dP_f + dP_pin)
   - r 越大（歧管越深）→ V_man 越小 → x_mal 越小 → 分配越均匀 → U 越小；
   - w、n 越大 → 通道压降越大 → x_mal 越小 → 分配越均匀（针肋的混合/均匀化收益）。
2) 局部强化均匀性（针肋宽度比 w、排数 n 主导）
   - 针肋增强局部换热、降低热点；但过稀（w→0）或过密（w→0.3、n 过大）都会产生
     局部流动扰动/流量不均 → 局部热点增多 → U 上升；
   - 故 U 随 w、n 呈非单调（w≈0.2、n≈4 附近取极小），用二次项刻画：
        x_w = (w - w_opt)^2,  x_n = (n - n_opt)^2

总模型（线性于特征，系数由附件2数据最小二乘标定）：
    U* = c0 + c1*x_mal + c2*x_w + c3*x_n + c4*n + c5*(r - r_opt)^2
    - c4*n     ：针肋排数线性驱动的局部扰动/分配不均
    - c5*(r-r_opt)^2：歧管存在最优深度（r_opt=3.5），过浅分配不均、过深流动扰动

复用压降模型（PressureDropModel）的几何与流动假设：
    N_ch=100, w_ch=0.08mm, H_ch 固定 0.5mm, M=4, fin_per_row=3, W_man=0.3mm。

作者：Codex（2026-08）
"""
from __future__ import annotations
from typing import Dict
from pressure_drop_model import PressureDropModel

class NonuniformityModel:
    def __init__(self, geo=None, water=None, conditions=None,
                 w_opt: float = 0.2, n_opt: float = 4.0, r_opt: float = 3.5):
        self.pdm = PressureDropModel(geo, water, conditions)
        self.w_opt = w_opt          # 局部强化最优针肋宽度比（标定）
        self.n_opt = n_opt          # 局部强化最优针肋排数（标定）
        self.r_opt = r_opt          # 流量分配最优歧管深高比（标定）

    def features(self, w: float, r: float, n: int) -> Dict[str, float]:
        """计算非均匀性的三个特征量。物理约束 w=0 ⇒ n=0。"""
        if w <= 0:
            n = 0
        c = self.pdm.components(w, r, n)          # 复用压降模型分量
        rho = self.pdm.water["rho"]
        V_man = c["V_man"]
        dyn_man = 0.5 * rho * V_man * V_man       # 歧管动压
        dP_ch = c["dP_f"] + c["dP_pin"]           # 通道压降（沿程+针肋）
        x_mal = dyn_man / max(dP_ch, 1e-9)        # 分配不均匀数
        x_w = (w - self.w_opt) ** 2               # w 偏离最优的局部扰动
        x_n = (n - self.n_opt) ** 2               # n 偏离最优的局部扰动
        n_lin = float(n)                          # 排数线性项（扰动随排数增长）
        x_r2 = (r - self.r_opt) ** 2              # r 偏离最优歧管深度的分配恶化
        return dict(w=w, r=r, n=n, x_mal=x_mal, x_w=x_w, x_n=x_n,
                    n_lin=n_lin, x_r2=x_r2, dyn_man=dyn_man, dP_ch=dP_ch)

    def predict(self, w: float, r: float, n: int, coef) -> float:
        """coef = [c0,c1,c2,c3,c4,c5]：
        U* = c0 + c1*x_mal + c2*x_w + c3*x_n + c4*n + c5*(r-r_opt)^2"""
        f = self.features(w, r, n)
        return (coef[0] + coef[1]*f["x_mal"] + coef[2]*f["x_w"] + coef[3]*f["x_n"]
                + coef[4]*f["n_lin"] + coef[5]*f["x_r2"])

if __name__ == "__main__":
    import numpy as np
    m = NonuniformityModel()
    for (w, r, n) in [(0.0, 3.0, 0), (0.2, 3.5, 4), (0.2, 3.0, 10), (0.3, 4.5, 2)]:
        f = m.features(w, r, n)
        print((f"w={w:<4} r={r:<4} n={n:<3} | x_mal={f['x_mal']:.3f} "
               f"x_w={f['x_w']:.4f} x_n={f['x_n']:.1f} dyn_man={f['dyn_man']:.0f} dP_ch={f['dP_ch']:.0f}"))
