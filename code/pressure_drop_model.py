# -*- coding: utf-8 -*-
"""
压降模型 pressure_drop_model.py
================================
选题B 第一问：压降机理模型（参考思路 1.1.doc，多段串联）。

    ΔP = ΔP_manifold + ΔP_channel + ΔP_pin + ΔP_turn
    - ΔP_manifold : 歧管分配层损失（含摩擦+动压），与歧管深高比 r 相关，r 越大损失越小
    - ΔP_channel  : 微通道沿程阻力（Darcy-Weisbach）
    - ΔP_pin      : 针肋绕流形阻，与针肋宽度比 w、排数 n 相关
    - ΔP_turn     : 进出口/转弯局部损失

复用 ThermalResistanceModel 的几何与流动参数（同热阻模型假设）：
    N_ch=100, w_ch=0.08mm, H_ch(r)=H_tot/(1+r), M=4, fin_per_row=3, ...
物理约束：w=0 时 n=0（无针肋则无针肋阻力）。

无量纲压降：数据 P* 由附件2给出；本模型输出 ΔP（Pa），
由 run_pressure_drop.py 用最小二乘拟合 P*_pred = c0 + Σ c_i·ΔP_i 完成标定与验证。

作者：Codex（2026-08）
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict
from thermal_resistance_model import ThermalResistanceModel, Geometry, WATER, CONDITIONS

@dataclass
class PressureGeometry(Geometry):
    """压降模型几何假设：
    - 微通道层深度 H_ch 取固定值（r 只改变歧管层深度 H_man=r*H_ch）
      —— 这是让"P 随 r 增大而下降"（歧管越深→歧管压降越小）成立的关键假设；
    - W_man 为歧管分配槽宽度（较窄→歧管压降占主导，典型值，可标定）。"""
    W_man: float = 0.3e-3          # 歧管分配槽宽度 m（较窄，歧管压降占主导）
    L_man: float = 2.0e-3          # 歧管分配槽长度 m（典型值）
    K_turn: float = 1.0            # 进出口/转弯局部损失系数（标定量，由LS拟合替代）

    def H_ch(self, r: float) -> float:
        """压降模型假设：微通道层深度固定，歧管深高比 r 只影响歧管层 H_man=r*H_ch。"""
        return self.H_ch_ref

def cd_cylinder(Re_d: float) -> float:
    """圆柱横掠阻力系数 C_D（近似）：
    Re_d<1: 10/Re_d（Stokes）；Re_d>=1: 1 + 10/Re_d^0.8"""
    if Re_d <= 0:
        return 0.0
    if Re_d < 1.0:
        return min(10.0 / Re_d, 100.0)
    return 1.0 + 10.0 / Re_d ** 0.8

def friction_factor(Re: float) -> float:
    """沿程摩擦因子：层流 64/Re；湍流 Blasius"""
    if Re < 2300:
        return 64.0 / Re
    return 0.316 * Re ** (-0.25)

class PressureDropModel:
    def __init__(self, geo: PressureGeometry | None = None,
                 water: dict | None = None, conditions: dict | None = None):
        self.geo = geo or PressureGeometry()
        self.water = dict(WATER if water is None else water)
        self.cond = dict(CONDITIONS if conditions is None else conditions)
        self.trm = ThermalResistanceModel(self.geo, self.water, self.cond)

    def components(self, w: float, r: float, n: int) -> Dict[str, float]:
        """计算四项压降（Pa）与中间量。物理约束 w=0 ⇒ n=0。"""
        if w <= 0:
            n = 0
        g = self.geo
        rho = self.water["rho"]; mu = self.water["mu"]
        fp = self.trm.flow_params(r, w)          # V, D_h, Re, H, A_c, V_gap
        V = fp["V"]; D_h = fp["D_h"]; Re = fp["Re"]
        H = fp["H"]; A_c = fp["A_c"]; V_gap = fp["V_gap"]
        mdot = self.cond["mdot"]

        # 1) 微通道沿程（Darcy-Weisbach，通道全长 L）
        f = friction_factor(Re)
        dP_f = f * (g.L / D_h) * 0.5 * rho * V * V

        # 2) 针肋绕流形阻：每条通道总排数 = M*n 排 × fin_per_row 根
        if w > 0 and n > 0:
            d = g.d_pin(w)
            Re_d = rho * V_gap * d / mu
            C_D = cd_cylinder(Re_d)
            A_front = d * H                       # 单针肋迎风面积
            n_pins = g.M * n * g.fin_per_row      # 每条通道的针肋总数
            dP_pin = n_pins * C_D * 0.5 * rho * V_gap * V_gap * A_front / A_c
        else:
            Re_d = 0.0; C_D = 0.0; dP_pin = 0.0

        # 3) 歧管分配层：H_man = r*H_ch，V_man 随 r 增大而减小 → ΔP 减小
        H_man = r * H
        A_man = g.W_man * H_man
        V_man = mdot / (rho * A_man)
        D_man = 2.0 * g.W_man * H_man / (g.W_man + H_man)
        Re_man = rho * V_man * D_man / mu
        f_man = friction_factor(Re_man)
        dP_man = (f_man * (g.L_man / D_man) + 1.0) * 0.5 * rho * V_man * V_man

        # 4) 进出口/转弯局部损失
        dP_turn = g.K_turn * 0.5 * rho * V * V

        dP_total = dP_f + dP_pin + dP_man + dP_turn
        return dict(w=w, r=r, n=n,
                    dP_f=dP_f, dP_pin=dP_pin, dP_man=dP_man, dP_turn=dP_turn,
                    dP_total=dP_total, Re=Re, Re_d=Re_d, C_D=C_D,
                    V=V, V_gap=V_gap, D_h=D_h, H=H, V_man=V_man, H_man=H_man)

if __name__ == "__main__":
    m = PressureDropModel()
    for (w, r, n) in [(0.0, 3.0, 0), (0.2, 3.0, 10), (0.2, 3.5, 6), (0.2, 4.5, 2), (0.3, 4.5, 2)]:
        c = m.components(w, r, n)
        print((f"w={w:<4} r={r:<4} n={n:<3} | dP_f={c['dP_f']:8.1f} dP_pin={c['dP_pin']:7.1f} "
               f"dP_man={c['dP_man']:7.1f} dP_turn={c['dP_turn']:6.1f} | total={c['dP_total']:8.1f} Pa "
               f"Re={c['Re']:.0f} Re_d={c['Re_d']:.1f} C_D={c['C_D']:.2f} V_man={c['V_man']:.3f}"))
