# -*- coding: utf-8 -*-
"""
热阻模型 thermal_resistance_model.py
====================================
选题B（高性能芯片热管理系统优化）第一问：热阻机理模型。

采用参考思路（1.1.doc）的串联热阻分解：
    R_total = R_cond + R_conv + R_fluid
    - R_cond : 衬底（氮化铝）导热热阻  = delta_sub / (k_AlN * A_foot)
    - R_conv : 微通道壁面 + 针肋表面对流换热热阻 = 1 / (h_eff * A_eff)
    - R_fluid: 冷却液流体升温热阻（热容阻力）= 1 / (2*mdot*Cp)

对流换热系数：
    - 通道壁面 Nu：层流充分发展 4.36 / 过渡湍流 Gnielinski
    - 针肋 Nu：Zukauskas 单圆柱横掠关联式（分段系数，Pr 指数 0.37）
    - 圆柱针肋效率 eta_fin = tanh(mH)/(mH)，m=sqrt(4h/(k_fin*d))
    - 针肋阻塞/尾流效应：有效针肋面积 = 原始面积 * (1 - cb*(w/w_max)^p)，
      体现针肋过粗（w 接近 0.3）时填满流道、死区换热失效的机理；
      cb、p 为半经验常数，由附件2数据标定（默认 cb=0.5, p=8）。

几何假设（图 A1/A2 未给全尺寸，取典型微通道针肋散热器几何并参数化）：
    - 10mm x 10mm 散热区，N_ch 条并联微通道，通道宽 w_ch
    - 微通道层深 H_ch(r) = H_tot/(1+r)（总高度近似固定，歧管加深则微通道变浅）
    - 圆柱针肋：直径 d_pin = w*w_ch，高 H_ch，每排 fin_per_row 根，每歧管单元 n 排
    - M 个歧管单元沿流向排列

无量纲热阻（工程常用定义，与附件2数据带一致）：
    R* = R_total * mdot * Cp

作者：Codex（2026-08）  语言：中文注释
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

# ============================================================
# 1. 物性参数与工况（附件 1）
# ============================================================
WATER = dict(rho=998.2, Cp=4182.0, k=0.6, mu=0.001003)   # 去离子水
ALN   = dict(rho=3260.0, Cp=700.0,  k=200.0)             # 氮化铝（衬底/针肋）
CHIP  = dict(rho=19320.0, Cp=130.0, k=298.0)             # 芯片等效材料

CONDITIONS = dict(
    mdot=1e-3,          # 入口质量流量 1 g/s
    T_in=293.0,         # 入口温度 K
    p_out=0.0,          # 出口表压 Pa
    T_amb=293.0,        # 环境 K
    h_nat=10.0,         # 外侧自然对流 W/(m2 K)
    q_v=5e9,            # 芯片体热源强度 W/m3
)

# ============================================================
# 2. 几何参数（默认值 + 可覆盖）
# ============================================================
@dataclass
class Geometry:
    L: float = 10e-3              # 芯片散热区边长 m
    N_ch: int = 100               # 并联微通道数
    w_ch: float = 0.08e-3         # 通道宽度 m
    H_ch_ref: float = 0.5e-3      # 微通道层参考深度 m（r=r_ref 时）
    r_ref: float = 3.5            # 参考歧管深高比
    delta_sub: float = 0.2e-3     # 衬底（含芯片等效层）厚度 m
    M: int = 4                    # 沿流向歧管单元数
    fin_per_row: int = 3          # 每排针肋根数（沿通道宽度方向）
    k_fin: float = ALN["k"]       # 针肋/衬底导热系数 W/(m K)
    w_max: float = 0.3            # 针肋宽度比上限（阻塞修正用）
    exposure_power: float = 8.0   # 阻塞/暴露面积修正指数 p
    exposure_strength: float = 0.5  # 阻塞修正强度 cb（w=w_max 时暴露面积=1-cb）

    @property
    def A_foot(self) -> float:
        return self.L * self.L

    def H_ch(self, r: float) -> float:
        """微通道层深度：总高度近似固定 H_tot=H_ch_ref*(1+r_ref)，
        歧管加深（r 增大）则微通道变浅：H_ch = H_tot/(1+r)。"""
        H_tot = self.H_ch_ref * (1.0 + self.r_ref)
        return H_tot / (1.0 + r)

    def d_pin(self, w: float) -> float:
        """针肋直径 = 针肋宽度比 * 通道宽度"""
        return w * self.w_ch

    def fin_exposure(self, w: float) -> float:
        """针肋有效面积系数：针肋过粗填满流道时，尾流/死区使部分换热面积失效。
        f(w) = 1 - cb*(w/w_max)^p，w=0 时为 1；w=w_max 时为 1-cb（保留部分换热能力）。
        参数 cb（exposure_strength）、p（exposure_power）由附件2数据标定。"""
        if w <= 0:
            return 1.0
        return max(0.0, 1.0 - self.exposure_strength * (w / self.w_max) ** self.exposure_power)

# ============================================================
# 3. 无量纲数与 Nu 关联式
# ============================================================
def prandtl() -> float:
    return WATER["mu"] * WATER["Cp"] / WATER["k"]

def nu_base_channel(Re: float, Pr: float) -> float:
    """微通道壁面 Nu：
    - 层流（Re<2300）：充分发展、等热流 Nu≈4.36
    - 过渡/湍流：Gnielinski 关联式"""
    if Re < 2300:
        return 4.36
    f = (0.790 * math.log(Re) - 1.64) ** -2  # Petukhov 摩擦因子
    return (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0/3.0) - 1.0))

def nu_pin_crossflow(Re_d: float, Pr: float) -> float:
    """Zukauskas：单圆柱横掠流动 Nu（分段系数，Pr 指数 0.37）"""
    if Re_d < 0.4:
        return 0.989 * Re_d ** 0.330 * Pr ** 0.37
    elif Re_d < 4.0:
        return 0.911 * Re_d ** 0.385 * Pr ** 0.37
    elif Re_d < 40.0:
        return 0.683 * Re_d ** 0.466 * Pr ** 0.37
    elif Re_d < 4000.0:
        return 0.193 * Re_d ** 0.618 * Pr ** 0.37
    else:
        return 0.027 * Re_d ** 0.805 * Pr ** 0.37

def fin_efficiency_cylinder(h: float, d: float, H: float, k_fin: float) -> float:
    """圆柱针肋效率 eta = tanh(mH)/(mH)，m=sqrt(4h/(k_fin*d))"""
    if d <= 0 or H <= 0:
        return 1.0
    m = math.sqrt(4.0 * h / (k_fin * d))
    mL = m * H
    return math.tanh(mL) / mL if mL > 1e-9 else 1.0

# ============================================================
# 4. 热阻模型主类
# ============================================================
class ThermalResistanceModel:
    def __init__(self, geo: Geometry | None = None, water: dict | None = None,
                 conditions: dict | None = None):
        self.geo = geo or Geometry()
        self.water = dict(WATER if water is None else water)
        self.cond = dict(CONDITIONS if conditions is None else conditions)
        self.Pr = prandtl()

    # ---- 流动参数 ----
    def flow_params(self, r: float, w: float) -> Dict[str, float]:
        g = self.geo
        H = g.H_ch(r)
        wc = g.w_ch
        A_c = wc * H                                  # 单通道流通截面积
        mdot_ch = self.cond["mdot"] / g.N_ch          # 每通道质量流量
        V = mdot_ch / (self.water["rho"] * A_c)       # 通道内平均流速
        D_h = 2.0 * wc * H / (wc + H)                 # 水力直径
        Re = self.water["rho"] * V * D_h / self.water["mu"]
        V_gap = V / (1.0 - w) if w < 1.0 else V       # 针肋间隙流速
        return dict(H=H, A_c=A_c, V=V, D_h=D_h, Re=Re, V_gap=V_gap, mdot_ch=mdot_ch)

    # ---- 换热面积（含针肋暴露修正）----
    def areas(self, w: float, r: float, n: int) -> Dict[str, float]:
        g = self.geo
        H = g.H_ch(r)
        L_unit = g.L / g.M
        A_wall_unit = g.w_ch * L_unit + 2.0 * H * L_unit   # 底壁 + 两侧壁（顶面为歧管盖）
        A_fin_unit = (n * g.fin_per_row * math.pi * g.d_pin(w) * H
                      * g.fin_exposure(w))                  # 针肋侧表面 × 暴露修正
        A_wall = g.N_ch * g.M * A_wall_unit
        A_fin  = g.N_ch * g.M * A_fin_unit
        return dict(A_wall=A_wall, A_fin=A_fin, A_eff=A_wall + A_fin,
                    A_wall_unit=A_wall_unit, A_fin_unit=A_fin_unit)

    # ---- 对流系数 ----
    def conv_coeffs(self, w: float, r: float, n: int) -> Dict[str, float]:
        g = self.geo
        fp = self.flow_params(r, w)
        ar = self.areas(w, r, n)
        kf = self.water["k"]
        h_wall = nu_base_channel(fp["Re"], self.Pr) * kf / fp["D_h"]
        out = dict(**fp, **ar)
        if w <= 0 or n <= 0:
            out.update(h_wall=h_wall, h_pin=0.0, eta_fin=1.0, h_eff=h_wall, Re_d=0.0)
            return out
        d = g.d_pin(w)
        Re_d = self.water["rho"] * fp["V_gap"] * d / self.water["mu"]
        Nu_d = nu_pin_crossflow(Re_d, self.Pr)
        h_pin = Nu_d * kf / d
        eta = fin_efficiency_cylinder(h_pin, d, fp["H"], g.k_fin)
        h_eff = (h_wall * ar["A_wall"] + eta * h_pin * ar["A_fin"]) / ar["A_eff"] \
            if ar["A_eff"] > 0 else h_wall
        out.update(h_wall=h_wall, h_pin=h_pin, eta_fin=eta, h_eff=h_eff, Re_d=Re_d)
        return out

    # ---- 三项热阻 ----
    def thermal_resistances(self, w: float, r: float, n: int) -> Dict[str, float]:
        """返回 R_cond, R_conv, R_fluid, R_total（K/W）及中间量。
        物理约束：w=0 时强制 n=0。"""
        if w <= 0:
            n = 0
        g = self.geo
        cc = self.conv_coeffs(w, r, n)
        R_cond = g.delta_sub / (g.k_fin * g.A_foot)
        R_conv = 1.0 / (cc["h_eff"] * cc["A_eff"])
        R_fluid = 1.0 / (2.0 * self.cond["mdot"] * self.water["Cp"])
        R_total = R_cond + R_conv + R_fluid
        out = dict(w=w, r=r, n=n,
                   R_cond=R_cond, R_conv=R_conv, R_fluid=R_fluid, R_total=R_total,
                   **cc)
        return out

    # ---- 无量纲热阻 ----
    def dimensionless_R(self, w: float, r: float, n: int, ref: str = "mdotCp") -> float:
        """R* 无量纲热阻：
        - 'mdotCp'（默认，工程常用定义）: R* = R_total * mdot * Cp
        - 'fluid' : R* = R_total / R_fluid = R_total*2*mdot*Cp
        - 'cond'  : R* = R_total / R_cond
        - float   : 自定义参考值"""
        R_total = self.thermal_resistances(w, r, n)["R_total"]
        if ref == "mdotCp":
            return R_total * self.cond["mdot"] * self.water["Cp"]
        if ref == "fluid":
            return R_total * 2.0 * self.cond["mdot"] * self.water["Cp"]
        if ref == "cond":
            return R_total / self.thermal_resistances(w, r, n)["R_cond"]
        return R_total / float(ref)

# ============================================================
# 5. 快捷入口
# ============================================================
if __name__ == "__main__":
    m = ThermalResistanceModel()
    print(f"Pr = {m.Pr:.3f}, A_foot = {m.geo.A_foot*1e4:.2f} cm2")
    print("w r  n | R_cond R_conv R_fluid R_total | R* (=R_total*mdot*Cp)")
    for (w, r, n) in [(0.0, 3.0, 0), (0.1, 3.5, 4), (0.2, 3.5, 4),
                      (0.2, 3.0, 10), (0.2, 4.5, 2), (0.3, 4.5, 4)]:
        d = m.thermal_resistances(w, r, n)
        print((f"{w:<4} {r:<4} {n:<3}| {d['R_cond']:.4f} {d['R_conv']:.4f} "
               f"{d['R_fluid']:.4f} {d['R_total']:.4f} | {d['R_total']*m.cond['mdot']*m.water['Cp']:.4f}"))
