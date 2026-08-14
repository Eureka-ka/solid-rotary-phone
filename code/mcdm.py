# -*- coding: utf-8 -*-
"""
多准则决策工具 mcdm.py
========================
- 熵权法：数据驱动客观赋权（指标越小越好，成本型）
- TOPSIS：距理想点/负理想点距离的接近度排序
- 灰色关联度分析：与理想解的关联度
"""
from __future__ import annotations
import numpy as np

def entropy_weights(X: np.ndarray) -> np.ndarray:
    """熵权法：X(n,m) 原始决策矩阵，指标均为成本型（越小越好）。
    返回权重 w(m)，和为 1。"""
    X = np.asarray(X, dtype=float)
    span = X.max(axis=0) - X.min(axis=0) + 1e-12
    Xn = (X.max(axis=0) - X) / span          # 正向化：越大越好
    P = Xn / (Xn.sum(axis=0) + 1e-12)
    k = 1.0 / np.log(len(X))
    e = -k * np.sum(P * np.log(P + 1e-12), axis=0)   # 信息熵
    w = (1.0 - e) / np.sum(1.0 - e)
    return w

def topsis_closeness(X: np.ndarray, w: np.ndarray, norm: str = "minmax") -> np.ndarray:
    """TOPSIS：X(n,m) 决策矩阵（成本型，越小越好），w 权重。返回接近度 C（越大越好）。
    norm='minmax'：各指标 min-max 到[0,1]（每个指标范围等权参与，权重才真正起作用）；
    norm='vector'：向量归一化（标准 TOPSIS）。"""
    X = np.asarray(X, dtype=float)
    if norm == "minmax":
        span = X.max(axis=0) - X.min(axis=0) + 1e-12
        Z = (X - X.min(axis=0)) / span               # [0,1]，0=该指标最优
    else:
        Z = X / np.sqrt(np.sum(X * X, axis=0) + 1e-12)
    V = Z * w
    ideal = V.min(axis=0)                            # 成本型：理想=最小
    nideal = V.max(axis=0)
    Dp = np.sqrt(np.sum((V - ideal) ** 2, axis=1))
    Dn = np.sqrt(np.sum((V - nideal) ** 2, axis=1))
    return Dn / (Dp + Dn + 1e-12)

def grey_relational_grade(X: np.ndarray, w: np.ndarray, xi: float = 0.5) -> np.ndarray:
    """灰色关联度：参考序列=各指标最小值（成本型理想解）。返回关联度（越大越好）。"""
    X = np.asarray(X, dtype=float)
    ref = X.min(axis=0)
    delta = np.abs(X - ref)
    dmin = delta.min(); dmax = delta.max()
    gamma = (dmin + xi * dmax) / (delta + xi * dmax)
    return np.sum(gamma * w, axis=1)

if __name__ == "__main__":
    X = np.array([[0.7, 0.1, 0.8], [0.6, 0.2, 0.7], [0.5, 0.3, 0.6], [0.4, 0.4, 0.5]])
    w = entropy_weights(X)
    print("熵权:", np.round(w, 3))
    print("TOPSIS C:", np.round(topsis_closeness(X, w), 3))
    print("灰色关联度:", np.round(grey_relational_grade(X, w), 3))
