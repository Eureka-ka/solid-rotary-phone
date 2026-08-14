# -*- coding: utf-8 -*-
"""
run_surrogate.py — 高斯过程回归代理模型主程序
1) 加载附件2（84 组）
2) 特征 (w,r,n) 归一化到[0,1]；目标 R、P、U 各建一个 GPR（RBF-ARD 核）
3) 超参数由对数边际似然最大化估计（坐标下降+重启）
4) 评估：训练 R²、闭式 LOOCV（R²_CV/RMSE/MAE/平均相对误差）、残差分析
5) 物理一致性检查：代理模型是否复现 Q1 的影响规律
6) 输出 CSV、指标、图
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from surrogate_gpr import GaussianProcessRegressor

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
DATA_XLSX = [p for p in glob.glob(r"E:\数学建模大赛\选题D\选题B\附件\*.xlsx")
                 if not os.path.basename(p).startswith("~$")][0]  # 排除Excel锁文件

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
print(f"样本数: {len(df)}")

# 特征归一化到[0,1]
X_raw = df[["w", "r", "n"]].values.astype(float)
xmin, xmax = X_raw.min(axis=0), X_raw.max(axis=0)
X = (X_raw - xmin) / (xmax - xmin)
targets = ["R", "P", "U"]

models = {}
summary = {}
train_preds = {}
loocv_preds = {}
for t in targets:
    y = df[t].values.astype(float)
    gp = GaussianProcessRegressor().fit(X, y, n_restarts=10)
    models[t] = gp
    # 训练预测
    mu_tr, _ = gp.predict(X)
    # LOOCV
    mu_lo, sd_lo = gp.loocv()
    def r2(yy, yp): return 1 - np.sum((yy - yp) ** 2) / np.sum((yy - yy.mean()) ** 2)
    def rmse(yy, yp): return float(np.sqrt(np.mean((yy - yp) ** 2)))
    def mae(yy, yp): return float(np.mean(np.abs(yy - yp)))
    summary[t] = dict(
        R2_train=round(r2(y, mu_tr), 4),
        R2_loocv=round(r2(y, mu_lo), 4),
        RMSE_loocv=round(rmse(y, mu_lo), 5),
        MAE_loocv=round(mae(y, mu_lo), 5),
        MRE_loocv_pct=round(float(np.mean(np.abs((y - mu_lo) / y))) * 100, 2),
        hp=gp.hyperparameters,
    )
    train_preds[t] = mu_tr
    loocv_preds[t] = mu_lo

print("\n================ 高斯过程回归（RBF-ARD 核）验证 ================")
for t in targets:
    s = summary[t]
    print(f"\n【{t}】")
    print(f"  训练 R² = {s['R2_train']} | LOOCV R² = {s['R2_loocv']} | "
          f"RMSE = {s['RMSE_loocv']} | MAE = {s['MAE_loocv']} | 平均相对误差 = {s['MRE_loocv_pct']}%")
    print(f"  超参数: sf={s['hp']['sf']:.3f}, l_w={s['hp']['l_w']:.3f}, l_r={s['hp']['l_r']:.3f}, "
          f"l_n={s['hp']['l_n']:.3f}, sn={s['hp']['sn']:.4f}")

# 物理一致性检查：在 84 个设计点上的预测均值按 w/r/n 分组，检查单调性方向
print("\n================ 物理一致性检查（Q1 规律复现） ================")
grid_pred = {t: models[t].predict(X)[0] for t in targets}
gpd = df[["w", "r", "n"]].copy()
for t in targets:
    gpd[t + "_gp"] = grid_pred[t]
checks = []
means_w = gpd.groupby("w").mean()
means_r = gpd.groupby("r").mean()
means_n = gpd.groupby("n").mean()

def check(name, ok):
    checks.append((name, ok)); print(f"  {'✅' if ok else '❌'} {name}")

# R: w 在0.2最小、n递减、r递增
check("R 随 w 先降后升（0.2最小）", means_w["R_gp"].idxmin() == 0.2)
check("R 随 n 递减", means_n["R_gp"].is_monotonic_decreasing)
check("R 随 r 总体递增", means_r["R_gp"].iloc[-1] > means_r["R_gp"].iloc[0])
# P: w递增、r递减、n递增
check("P 随 w 递增", means_w["P_gp"].is_monotonic_increasing)
check("P 随 r 递减", means_r["P_gp"].is_monotonic_decreasing)
check("P 随 n 递增", means_n["P_gp"].is_monotonic_increasing)
# U: w在0.2最小、n在4最小
check("U 随 w 先降后升（0.2最小）", means_w["U_gp"].idxmin() == 0.2)
check("U 随 n 先降后升（4最小）", means_n["U_gp"].idxmin() == 4)

# 保存
md = df[["id", "w", "r", "n", "R", "P", "U"]].copy()
for t in targets:
    md[t + "_train"] = train_preds[t]
    md[t + "_loocv"] = loocv_preds[t]
csv_path = unlocked_path(os.path.join(OUT, "surrogate_gpr_predictions.csv"))
md.to_csv(csv_path, index=False, encoding="utf-8-sig")

txt_path = unlocked_path(os.path.join(OUT, "surrogate_gpr_metrics.txt"))
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("高斯过程回归代理模型验证报告（RBF-ARD 核 + LOOCV）\n" + "=" * 50 + "\n")
    for t in targets:
        s = summary[t]
        f.write(f"\n[{t}]\n")
        f.write(f"  训练 R²: {s['R2_train']}\n")
        f.write(f"  LOOCV R²: {s['R2_loocv']}\n")
        f.write(f"  LOOCV RMSE: {s['RMSE_loocv']}\n")
        f.write(f"  LOOCV MAE: {s['MAE_loocv']}\n")
        f.write(f"  LOOCV 平均相对误差%: {s['MRE_loocv_pct']}\n")
        f.write(f"  超参数: {s['hp']}\n")
    f.write("\n物理一致性检查\n")
    for name, ok in checks:
        f.write(f"  {'OK ' if ok else 'FAIL'} {name}\n")
print(f"\n已保存: {csv_path}")
print(f"已保存: {txt_path}")

# 绘图
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

    # 图1：LOOCV 预测 vs 真值（含不确定性误差棒）
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, t in zip(axes, targets):
        y = df[t].values
        mu, sd = models[t].predict(X)
        lo, hi = loocv_preds[t] - sd, loocv_preds[t] + sd
        ax.errorbar(y, loocv_preds[t], yerr=sd, fmt="o", ms=4, alpha=0.6, color="steelblue", capsize=2)
        lim = [min(y.min(), loocv_preds[t].min()), max(y.max(), loocv_preds[t].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"数据 {t}"); ax.set_ylabel(f"LOOCV 预测 {t}")
        ax.set_title(f"{t}: LOOCV R²={summary[t]['R2_loocv']}")
        ax.grid(alpha=0.3)
    fig.suptitle("高斯过程回归 LOOCV 预测 vs 真值（误差棒=预测±1σ）", y=1.02)
    plt.tight_layout()
    f1 = os.path.join(OUT, "surrogate_gpr_loocv.png")
    plt.savefig(f1, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f1)

    # 图2：残差 vs 各参数
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    params = ["w", "r", "n"]
    for i, t in enumerate(targets):
        res = df[t].values - loocv_preds[t]
        for j, pm in enumerate(params):
            ax = axes[i, j]
            ax.scatter(df[pm], res, s=18, alpha=0.7, color="tab:green")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel(pm); ax.set_ylabel("残差")
            if i == 0: ax.set_title(f"残差 vs {pm}")
        axes[i, 0].set_ylabel(f"{t} 残差")
    fig.suptitle("LOOCV 残差分析（残差 vs w/r/n）", y=1.0)
    plt.tight_layout()
    f2 = os.path.join(OUT, "surrogate_gpr_residuals.png")
    plt.savefig(f2, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f2)

    # 图3：代理模型趋势 vs 数据趋势（按参数平均）
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for i, t in enumerate(targets):
        for j, (pm, lbl) in enumerate(zip(params, ["针肋宽度比 w", "歧管深高比 r", "针肋排数 n"])):
            ax = axes[i, j]
            g_data = df.groupby(pm)[t].mean()
            g_gp = gpd.groupby(pm)[t + "_gp"].mean()
            idx = g_data.index.astype(float) if pm != "n" else g_data.index.astype(int)
            ax.plot(idx, g_data.values, "o-", color="crimson", label="数据")
            ax.plot(idx, g_gp.values, "s--", color="steelblue", label="GP")
            ax.set_xlabel(lbl); ax.grid(alpha=0.3)
            if j == 0: ax.set_ylabel(f"{t}")
            if i == 0: ax.legend(fontsize=8)
    fig.suptitle("代理模型趋势 vs 附件2数据（按参数平均）", y=1.0)
    plt.tight_layout()
    f3 = os.path.join(OUT, "surrogate_gpr_trends.png")
    plt.savefig(f3, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f3)
except Exception as e:
    print("绘图失败（不影响结果）:", e)
