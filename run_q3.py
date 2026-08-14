# -*- coding: utf-8 -*-
"""
run_q3.py — Q3 多目标优化主程序（Pareto + 多准则决策）
=====================================================
1) 训练 Q2 高斯过程代理模型 (R,P,U)
2) 候选设计：穷举网格（w x r x 6档n）+ 非支配排序 → 精确 Pareto 前沿
   并用自实现 NSGA-II 求前沿做交叉验证
3) 决策：熵权法客观赋权（数据驱动）→ TOPSIS 选综合最优；等权 TOPSIS / 灰色关联度对比
4) 验证：最优解回代 GP（±σ），并与附件2 最近样本点对比
5) 输出：CSV + Pareto 3D/投影图 + 权重图 + docs/Q3 总结
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from surrogate_gpr import GaussianProcessRegressor
from nsga2 import NSGA2, pareto_front_indices
from mcdm import entropy_weights, topsis_closeness, grey_relational_grade

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
DATA_XLSX = [p for p in glob.glob(r"E:\数学建模大赛\选题D\选题B\附件\*.xlsx")
             if not os.path.basename(p).startswith("~$")][0]

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

df = pd.read_excel(DATA_XLSX, header=1)
df.columns = ["id", "w", "r", "n", "R", "P", "U"]
X_raw = df[["w", "r", "n"]].values.astype(float)
xmin, xmax = X_raw.min(axis=0), X_raw.max(axis=0)

def norm(X):
    return (np.atleast_2d(X) - xmin) / (xmax - xmin)

# ---------- 1. 训练 GP ----------
print("训练高斯过程代理模型 ...")
gps = {}
for t in ["R", "P", "U"]:
    gps[t] = GaussianProcessRegressor().fit(norm(X_raw), df[t].values.astype(float), n_restarts=10)
    print(f"  {t}: LOOCV 超参数就绪")

def gpr_predict(t, X):
    """X(n,3) 原始尺度 → 预测均值(n,) 与标准差(n,)"""
    mu, sd = gps[t].predict(norm(X))
    return np.asarray(mu), np.asarray(sd)

def objectives(X):
    """X(n,3) 原始尺度 → F(n,3) = [R,P,U]"""
    X = np.asarray(X, dtype=float)
    R, _ = gpr_predict("R", X); P, _ = gpr_predict("P", X); U, _ = gpr_predict("U", X)
    return np.column_stack([R, P, U])

# ---------- 2. 网格候选 + 精确 Pareto 前沿 ----------
print("\n生成网格候选并求 Pareto 前沿 ...")
w_grid = np.round(np.concatenate([[0.0], np.linspace(0.1, 0.3, 41)]), 4)  # 数据覆盖区 {0}∪[0.1,0.3]
r_grid = np.round(np.linspace(3, 4.5, 31), 3)
n_set = np.array([0, 2, 4, 6, 8, 10], dtype=int)
gW, gR, gN = np.meshgrid(w_grid, r_grid, n_set, indexing="ij")
G = np.column_stack([gW.ravel(), gR.ravel(), gN.ravel()])
# 有效组合：无针肋(w=0,n=0) 或 有针肋(w>=0.1, n>0) —— 排除 w 在 (0,0.1) 的外推假象区
valid = ((G[:, 2] == 0) & (G[:, 0] == 0)) | ((G[:, 2] > 0) & (G[:, 0] >= 0.1))
G = G[valid]
FG = objectives(G)
pf_grid = pareto_front_indices(FG)
Gp, Fp = G[pf_grid], FG[pf_grid]
print(f"  候选数 {len(G)}，Pareto 前沿点 {len(Gp)}")

# ---------- 3. NSGA-II 交叉验证 ----------
print("运行 NSGA-II ...")
ns = NSGA2(bounds=[(0.1, 0.3), (3.0, 4.5), (0.0, 10.0)], pop_size=200, n_gen=600, pm=0.15, seed=7)
Xns, Fns = ns.run(objectives)
# NSGA-II 与网格前沿的一致性：每个 NSGA-II 解到网格前沿的最短目标距离
# 直接用向量化计算
D = np.sqrt(((Fns[:, None, :] - Fp[None, :, :]) ** 2).sum(axis=2))  # (Nns, Np)
min_d = D.min(axis=1)
print(f"  NSGA-II 前沿 {len(Fns)} 点；到网格前沿最短距离: 均值={min_d.mean():.5f}, "
      f"中位数={np.median(min_d):.5f}, 最大={min_d.max():.5f}（≈0 即一致）")

# ---------- 4. 熵权法（数据驱动）----------
print("\n熵权法（基于附件2 84 组样本）:")
W_ent = entropy_weights(df[["R", "P", "U"]].values)
print(f"  w_R={W_ent[0]:.3f}, w_P={W_ent[1]:.3f}, w_U={W_ent[2]:.3f}")

# ---------- 5. 多准则决策（在网格 Pareto 前沿上）----------
print("\n多准则决策（TOPSIS / 灰色关联度）:")
C_ent = topsis_closeness(Fp, W_ent)              # 熵权-TOPSIS（主）
C_eq = topsis_closeness(Fp, np.ones(3) / 3)      # 等权-TOPSIS（对比）
Gg = grey_relational_grade(Fp, W_ent)            # 熵权-灰色关联（对比）
i_opt = int(np.argmax(C_ent))
best = Gp[i_opt]
print(f"  熵权-TOPSIS 最优: w={best[0]:.3f}, r={best[1]:.3f}, n={int(best[2])}  "
      f"(C={C_ent[i_opt]:.4f})")
print(f"  等权-TOPSIS 最优: w={Gp[np.argmax(C_eq)][0]:.3f}, r={Gp[np.argmax(C_eq)][1]:.3f}, "
      f"n={int(Gp[np.argmax(C_eq)][2])}")
print(f"  熵权-灰色关联最优: w={Gp[np.argmax(Gg)][0]:.3f}, r={Gp[np.argmax(Gg)][1]:.3f}, "
      f"n={int(Gp[np.argmax(Gg)][2])}")

# ---------- 6. 最优解验证 ----------
Rmu, Rsd = gpr_predict("R", best.reshape(1, -1))
Pmu, Psd = gpr_predict("P", best.reshape(1, -1))
Umu, Usd = gpr_predict("U", best.reshape(1, -1))
print(f"\n最优方案 (w={best[0]:.3f}, r={best[1]:.3f}, n={int(best[2])}) 的 GP 预测:")
print(f"  R = {Rmu[0]:.4f} ± {Rsd[0]:.4f}")
print(f"  P = {Pmu[0]:.4f} ± {Psd[0]:.4f}")
print(f"  U = {Umu[0]:.4f} ± {Usd[0]:.4f}")
# 最近样本点
dist = np.sqrt(((df[["w", "r", "n"]].values - best) ** 2).sum(axis=1))
j = int(np.argmin(dist))
print(f"  附件2 最近样本 #{int(df.id[j])}: w={df.w[j]}, r={df.r[j]}, n={int(df.n[j])} "
      f"→ R={df.R[j]:.4f}, P={df.P[j]:.4f}, U={df.U[j]:.4f} (距离 {dist[j]:.4f})")

# ---------- 7. 保存 ----------
pd.DataFrame(np.column_stack([Gp, Fp]), columns=["w", "r", "n", "R", "P", "U"]).to_csv(
    unlocked_path(os.path.join(OUT, "q3_pareto_grid.csv")), index=False, encoding="utf-8-sig")
pd.DataFrame(np.column_stack([Xns, Fns]), columns=["w", "r", "n", "R", "P", "U"]).to_csv(
    unlocked_path(os.path.join(OUT, "q3_pareto_nsga2.csv")), index=False, encoding="utf-8-sig")
rank = pd.DataFrame({
    "w": Gp[:, 0], "r": Gp[:, 1], "n": Gp[:, 2].astype(int),
    "R": Fp[:, 0], "P": Fp[:, 1], "U": Fp[:, 2],
    "C_entropy_topsis": C_ent, "C_equal_topsis": C_eq, "grey_grade": Gg,
}).sort_values("C_entropy_topsis", ascending=False)
rank.to_csv(unlocked_path(os.path.join(OUT, "q3_mcdm_ranking.csv")), index=False, encoding="utf-8-sig")
opt_row = rank.iloc[0]
with open(unlocked_path(os.path.join(OUT, "q3_optimum.txt")), "w", encoding="utf-8") as f:
    f.write("Q3 综合最优设计方案（熵权-TOPSIS）\n" + "=" * 40 + "\n")
    f.write(f"w* = {best[0]:.3f}, r* = {best[1]:.3f}, n* = {int(best[2])}\n")
    f.write(f"GP 预测: R = {Rmu[0]:.4f} ± {Rsd[0]:.4f}\n")
    f.write(f"         P = {Pmu[0]:.4f} ± {Psd[0]:.4f}\n")
    f.write(f"         U = {Umu[0]:.4f} ± {Usd[0]:.4f}\n")
    f.write(f"熵权: w_R={W_ent[0]:.3f}, w_P={W_ent[1]:.3f}, w_U={W_ent[2]:.3f}\n")
    f.write(f"NSGA-II 前沿 {len(Fns)} 点；到网格前沿最短距离均值 {min_d.mean():.5f}\n")
print("\n已保存 CSV 与最优方案文本。")

# ---------- 8. 绘图 ----------
try:
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

    # 3D Pareto
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(FG[:, 0], FG[:, 1], FG[:, 2], s=6, alpha=0.25, c="0.8", label="全部候选")
    ax.scatter(Fp[:, 0], Fp[:, 1], Fp[:, 2], s=14, alpha=0.8, c="steelblue", label="Pareto 前沿")
    ax.scatter([Rmu[0]], [Pmu[0]], [Umu[0]], s=90, c="crimson", marker="*", label="熵权-TOPSIS 最优")
    ax.set_xlabel("R"); ax.set_ylabel("P"); ax.set_zlabel("U")
    ax.set_title("Q3 Pareto 前沿（3D）")
    ax.legend(fontsize=9)
    f1 = os.path.join(OUT, "q3_pareto_3d.png")
    plt.savefig(f1, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f1)

    # 2D 投影
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    pairs = [("R", "P", 0, 1), ("R", "U", 0, 2), ("P", "U", 1, 2)]
    for ax, (xlab, ylab, i, j), in zip(axes, pairs):
        ax.scatter(FG[:, i], FG[:, j], s=6, alpha=0.2, c="0.8")
        ax.scatter(Fp[:, i], Fp[:, j], s=12, alpha=0.8, c="steelblue")
        ax.scatter([Fp[i_opt, i]], [Fp[i_opt, j]], s=80, c="crimson", marker="*", zorder=5)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        ax.set_title(f"Pareto 投影（{xlab}-{ylab}）")
    fig.suptitle("Q3 Pareto 前沿二维投影（★=熵权-TOPSIS 最优）", y=1.02)
    plt.tight_layout()
    f2 = os.path.join(OUT, "q3_pareto_projections.png")
    plt.savefig(f2, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f2)

    # 熵权 + 决策对比
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(["R", "P", "U"], W_ent, color=["#9ecae1", "#fdae6b", "#a1d99b"])
    for k, v in enumerate(W_ent):
        axes[0].text(k, v + 0.01, f"{v:.3f}", ha="center")
    axes[0].set_ylabel("权重"); axes[0].set_title("熵权法权重（数据驱动）"); axes[0].grid(axis="y", alpha=0.3)
    top = rank.head(15)
    axes[1].barh(np.arange(len(top)), top["C_entropy_topsis"].values[::-1], color="steelblue")
    axes[1].set_yticks(np.arange(len(top)))
    axes[1].set_yticklabels([f"w={a:.2f},r={b:.2f},n={int(c)}" for a, b, c in
                             top[["w", "r", "n"]].values[::-1]], fontsize=7)
    axes[1].set_xlabel("TOPSIS 接近度 C"); axes[1].set_title("Pareto 前沿 TOPSIS 排序 Top15")
    axes[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    f3 = os.path.join(OUT, "q3_mcdm.png")
    plt.savefig(f3, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f3)
except Exception as e:
    print("绘图失败（不影响结果）:", e)
