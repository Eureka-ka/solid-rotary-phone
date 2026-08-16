# -*- coding: utf-8 -*-
"""
plot_q4.py — Q4 可视化（三元分区图 / 后悔与膝点 / 雷达图 / 场景损失）
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\选题B\outputs"
Fp = pd.read_csv(os.path.join(OUT, "q3_pareto_grid.csv"))
F = Fp[["R", "P", "U"]].values; X = Fp[["w", "r", "n"]].values
Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
N = len(F)

# 权重网格
step = 0.02
W = []
for a in np.round(np.arange(0, 1 + step, step), 4):
    for b in np.round(np.arange(0, 1 - a + step, step), 4):
        g = round(1 - a - b, 4)
        if g >= 0: W.append((a, b, g))
W = np.array(W)
S = W[:, 0][:, None] * Fn[:, 0][None, :] + W[:, 1][:, None] * Fn[:, 1][None, :] + W[:, 2][:, None] * Fn[:, 2][None, :]
opt_idx = S.argmin(1)
opt_score = S.min(1)
worst_regret = (S.T - opt_score[None, :]).max(1)
dist_utopia = np.sqrt((Fn ** 2).sum(1))
i_m2 = int(np.argmin(worst_regret)); i_m4 = int(np.argmin(dist_utopia))

# ============ 图1：三元稳定性分区图 ============
# 重心坐标 → 平面坐标
a, b, c = W[:, 0], W[:, 1], W[:, 2]
px = b + 0.5 * c
py = (np.sqrt(3) / 2) * c
# 每个最优设计一个标签，取频次前K个
uniq, counts = np.unique(opt_idx, return_counts=True)
K = 8
top = uniq[np.argsort(-counts)[:K]]
cmap = plt.cm.tab10
label_map = {d: k for k, d in enumerate(top)}
colors = np.array([label_map[i] if i in label_map else K for i in opt_idx])
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(px, py, c=colors, cmap=cmap, s=6, alpha=0.8)
# 三角形边界
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], "k-", lw=1)
ax.text(0, -0.06, "α(R)权重=1", ha="center")
ax.text(1, -0.06, "β(P)权重=1", ha="center")
ax.text(0.5, np.sqrt(3)/2 + 0.05, "γ(U)权重=1", ha="center")
# 图例
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(k), markersize=8,
                      label=f"w={X[d,0]:.2f}, r={X[d,1]:.2f}, n={int(X[d,2])} ({counts[np.where(uniq==d)[0][0]]/len(W)*100:.0f}%)")
           for k, d in enumerate(top)]
handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(K), markersize=8, label="其他(占比小)"))
ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_title("权重单纯形：最优方案稳定性分区图（1326 个网格权重）")
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
f1 = os.path.join(OUT, "q4_weight_simplex.png")
plt.savefig(f1, dpi=140, bbox_inches="tight"); plt.close()
print("saved", f1)

# ============ 图2：鲁棒性指标（最坏后悔 + 距理想点） ============
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
top10 = np.argsort(worst_regret)[:10]
axes[0].barh(np.arange(10)[::-1], worst_regret[top10][::-1], color="steelblue")
axes[0].set_yticks(np.arange(10)[::-1])
axes[0].set_yticklabels([f"w={X[i,0]:.2f},r={X[i,1]:.2f},n={int(X[i,2])}" for i in top10[::-1]], fontsize=8)
axes[0].set_xlabel("最坏后悔 max_w regret"); axes[0].set_title("方法2：最坏后悔最小的 Top10 设计")
axes[0].grid(axis="x", alpha=0.3)
top10d = np.argsort(dist_utopia)[:10]
axes[1].barh(np.arange(10)[::-1], dist_utopia[top10d][::-1], color="#fdae6b")
axes[1].set_yticks(np.arange(10)[::-1])
axes[1].set_yticklabels([f"w={X[i,0]:.2f},r={X[i,1]:.2f},n={int(X[i,2])}" for i in top10d[::-1]], fontsize=8)
axes[1].set_xlabel("距理想点距离"); axes[1].set_title("方法4：距理想点最近(膝点)的 Top10 设计")
axes[1].grid(axis="x", alpha=0.3)
plt.tight_layout()
f2 = os.path.join(OUT, "q4_robustness.png")
plt.savefig(f2, dpi=140, bbox_inches="tight"); plt.close()
print("saved", f2)

# ============ 图3：雷达图（归一化 R/P/U） ============
scenarios = {"高性能": (0.6, 0.2, 0.2), "节能": (0.2, 0.6, 0.2), "可靠性": (0.2, 0.2, 0.6), "均衡": (1/3, 1/3, 1/3)}
robust_i = i_m4   # 膝点为推荐鲁棒方案
labels = ["R(热阻)", "P(压降)", "U(非均匀)"]
angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
def radar(vals, name, color, ls="-", lw=1.6):
    v = np.concatenate([vals, [vals[0]]])
    ax.plot(angles, v, ls, color=color, lw=lw, label=name)
    ax.fill(angles, v, color=color, alpha=0.08)
radar(Fn[robust_i], f"鲁棒方案(膝点) w={X[robust_i,0]:.2f},r={X[robust_i,1]:.2f},n={int(X[robust_i,2])}", "crimson", "-", 2.2)
for k, (sname, wv) in enumerate(scenarios.items()):
    wv = np.array(wv)
    j = int(np.argmin(wv @ Fn.T))
    radar(Fn[j], f"{sname}专属最优 w={X[j,0]:.2f},n={int(X[j,2])}", plt.cm.tab10(k))
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1); ax.set_title("鲁棒方案 vs 场景专属最优（归一化指标，越小越好）")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
f3 = os.path.join(OUT, "q4_radar.png")
plt.savefig(f3, dpi=140, bbox_inches="tight"); plt.close()
print("saved", f3)

# ============ 图4：场景性能损失对比 ============
rows = []
for sname, wv in scenarios.items():
    wv = np.array(wv)
    sc = wv @ Fn.T
    j = int(np.argmin(sc))
    rows.append((sname, "膝点鲁棒", sc[robust_i] - sc[j]))
    rows.append((sname, "minimax鲁棒", sc[i_m2] - sc[j]))
d = pd.DataFrame(rows, columns=["场景", "方案", "损失"])
piv = d.pivot(index="场景", columns="方案", values="损失")[["膝点鲁棒", "minimax鲁棒"]]
ax = piv.plot(kind="bar", figsize=(9, 4.5), color=["#a1d99b", "#fdae6b"])
ax.set_ylabel("相对场景最优的得分损失"); ax.set_title("鲁棒方案在各场景下的性能损失（越小越好）")
ax.grid(axis="y", alpha=0.3); ax.legend()
plt.tight_layout()
f4 = os.path.join(OUT, "q4_scenario_loss.png")
plt.savefig(f4, dpi=140, bbox_inches="tight"); plt.close()
print("saved", f4)
