# -*- coding: utf-8 -*-
"""
run_q4.py — Q4 权重敏感性与鲁棒设计主程序
========================================
A. 场景权重：高性能(0.6,0.2,0.2) / 节能(0.2,0.6,0.2) / 可靠性(0.2,0.2,0.6) / 均衡(1/3,1/3,1/3)
B. 权重敏感性：单纯形网格采样（w1+w2+w3=1），每组权重用加权和（min-max 归一化）
   从 Pareto 前沿选最优 → 统计频次 → 稳定性分区图（三元图）
C. 鲁棒设计判据（按用户选择）：
   方法2：Minimax Regret —— x* = argmin_x max_w [F_w(x) - F_w(x*(w))]（最坏偏好下后悔最小）
   方法4：Pareto 前沿膝点 —— 最小距离法（距归一化理想点(0,0,0)最近的前沿点）
D. 优势论证：4 场景下鲁棒方案 vs 场景专属最优的性能损失；雷达图
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

def unlocked_path(path):
    if not os.path.exists(path):
        return path
    try:
        with open(path, "a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        i = 2
        while True:
            cand = f"{base}_v{i}{ext}"
            if not os.path.exists(cand):
                return cand
            try:
                with open(cand, "a", encoding="utf-8"):
                    pass
                return cand
            except PermissionError:
                i += 1

# ---------- 1. 加载 Pareto 前沿 ----------
Fp = pd.read_csv(os.path.join(OUT, "q3_pareto_grid.csv"))
F = Fp[["R", "P", "U"]].values
X = Fp[["w", "r", "n"]].values
N = len(F)
print(f"Pareto 前沿设计数: {N}")

# min-max 归一化（0=最优）
Fmin, Fmax = F.min(0), F.max(0)
Fn = (F - Fmin) / (Fmax - Fmin + 1e-12)

# ---------- 2. 权重单纯形网格采样 ----------
step = 0.02
weights = []
a_vals = np.round(np.arange(0, 1 + step, step), 4)
for a in a_vals:
    for b in np.round(np.arange(0, 1 - a + step, step), 4):
        g = round(1 - a - b, 4)
        if g < -1e-9: continue
        weights.append((a, b, max(g, 0.0)))
W = np.array(weights)
print(f"权重网格点数: {len(W)}")

# ---------- 3. 每组权重选最优（加权和） ----------
S = W[:, 0][:, None] * Fn[:, 0][None, :] + W[:, 1][:, None] * Fn[:, 1][None, :] + W[:, 2][:, None] * Fn[:, 2][None, :]
opt_idx = S.argmin(axis=1)               # 每个权重的最优设计下标
opt_score = S.min(axis=1)                # 每个权重的最优得分

# 频次统计
uniq, counts = np.unique(opt_idx, return_counts=True)
freq = counts / len(W)
order = np.argsort(-counts)
print("\n=== 各设计作为最优的频次（Top8）===")
for k in order[:8]:
    i = uniq[k]
    print(f"  (w={X[i,0]:.3f}, r={X[i,1]:.3f}, n={int(X[i,2])}): 频次 {counts[k]} ({freq[k]*100:.1f}%)  R={F[i,0]:.4f} P={F[i,1]:.4f} U={F[i,2]:.4f}")

# ---------- 4. 方法2：Minimax Regret ----------
regret = S.T - opt_score[None, :]        # (N, nW)：每个设计在各权重下的后悔
worst_regret = regret.max(axis=1)        # 每个设计的最坏后悔
i_m2 = int(np.argmin(worst_regret))
print("\n=== 方法2：Minimax Regret ===")
print(f"  鲁棒设计: (w={X[i_m2,0]:.3f}, r={X[i_m2,1]:.3f}, n={int(X[i_m2,2])})  最坏后悔={worst_regret[i_m2]:.4f}")
for k in np.argsort(worst_regret)[:5]:
    print(f"    (w={X[k,0]:.3f}, r={X[k,1]:.3f}, n={int(X[k,2])}) 最坏后悔={worst_regret[k]:.4f}")

# ---------- 5. 方法4：膝点（最小距离法，距理想点最近） ----------
dist_utopia = np.sqrt((Fn ** 2).sum(axis=1))
i_m4 = int(np.argmin(dist_utopia))
print("\n=== 方法4：膝点（最小距离法）===")
print(f"  膝点设计: (w={X[i_m4,0]:.3f}, r={X[i_m4,1]:.3f}, n={int(X[i_m4,2])})  距理想点={dist_utopia[i_m4]:.4f}")

# ---------- 6. 场景权重与优势论证 ----------
scenarios = {
    "高性能(AI/HPC)": (0.6, 0.2, 0.2),
    "节能(移动/低功耗)": (0.2, 0.6, 0.2),
    "可靠性(航天/工业)": (0.2, 0.2, 0.6),
    "均衡": (1/3, 1/3, 1/3),
}
print("\n=== 场景对比：鲁棒方案 vs 场景专属最优 ===")
rows = []
for sname, wv in scenarios.items():
    wv = np.array(wv)
    sc = wv @ Fn.T
    j = int(np.argmin(sc))
    # 鲁棒方案：取两种方法一致/接近者
    for tag, i in [("方法2", i_m2), ("方法4", i_m4)]:
        loss = sc[i] - sc[j]
        rows.append((sname, tag, X[i], F[i], X[j], F[j], loss))
        print(f"  [{sname}] {tag}鲁棒 (w={X[i,0]:.2f},r={X[i,1]:.2f},n={int(X[i,2])}) 得分={sc[i]:.4f} "
              f"| 场景最优 (w={X[j,0]:.2f},r={X[j,1]:.2f},n={int(X[j,2])}) 得分={sc[j]:.4f} | 损失={loss:.4f}")

# ---------- 7. 保存 ----------
# 权重-最优映射
wmap = pd.DataFrame({"alpha": W[:, 0], "beta": W[:, 1], "gamma": W[:, 2],
                     "opt_w": X[opt_idx, 0], "opt_r": X[opt_idx, 1], "opt_n": X[opt_idx, 2].astype(int)})
wmap.to_csv(unlocked_path(os.path.join(OUT, "q4_weight_map.csv")), index=False, encoding="utf-8-sig")
# 前沿各设计鲁棒性指标
rob = pd.DataFrame({"w": X[:, 0], "r": X[:, 1], "n": X[:, 2].astype(int),
                    "R": F[:, 0], "P": F[:, 1], "U": F[:, 2],
                    "worst_regret": worst_regret, "dist_utopia": dist_utopia,
                    "freq_optimal": np.array([freq[list(uniq).index(i)] if i in uniq else 0.0 for i in range(N)])})
rob.to_csv(unlocked_path(os.path.join(OUT, "q4_robustness_metrics.csv")), index=False, encoding="utf-8-sig")

with open(unlocked_path(os.path.join(OUT, "q4_robust_design.txt")), "w", encoding="utf-8") as f:
    f.write("Q4 鲁棒设计方案（网格权重采样）\n" + "=" * 46 + "\n")
    f.write(f"方法2 Minimax Regret: (w={X[i_m2,0]:.3f}, r={X[i_m2,1]:.3f}, n={int(X[i_m2,2])})  最坏后悔={worst_regret[i_m2]:.4f}\n")
    f.write(f"方法4 膝点(最小距离): (w={X[i_m4,0]:.3f}, r={X[i_m4,1]:.3f}, n={int(X[i_m4,2])})  距理想点={dist_utopia[i_m4]:.4f}\n")
    f.write(f"R/P/U: 方法2→{F[i_m2,0]:.4f}/{F[i_m2,1]:.4f}/{F[i_m2,2]:.4f}  方法4→{F[i_m4,0]:.4f}/{F[i_m4,1]:.4f}/{F[i_m4,2]:.4f}\n")
print("\n已保存 CSV 与鲁棒方案文本。")
