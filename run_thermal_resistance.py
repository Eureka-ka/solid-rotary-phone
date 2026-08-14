# -*- coding: utf-8 -*-
"""
run_thermal_resistance.py — 热阻模型主程序
===========================================
1) 加载附件2样本数据（84 组）
2) 用机理模型在相同设计网格上计算无量纲热阻 R*（默认 R* = R_total*mdot*Cp）
3) 仿射标定 R*_pred = c0 + c1*R*_model（最小二乘），吸收几何/归一化不确定性
4) 验证：Pearson/Spearman 相关系数、R^2、RMSE、MAE、平均相对误差
5) 输出：趋势对比表、验证指标、模型预测 CSV、三张图

输出目录：E:/选题B/outputs/
"""
from __future__ import annotations
import os, sys, glob, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from thermal_resistance_model import ThermalResistanceModel, Geometry

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

DATA_XLSX = [p for p in glob.glob(r"E:\数学建模大赛\选题D\选题B\附件\*.xlsx")
                 if not os.path.basename(p).startswith("~$")][0]  # 排除Excel锁文件

# ---------------- 1. 加载数据 ----------------
df = pd.read_excel(DATA_XLSX, header=1)
df.columns = ["id", "w", "r", "n", "R", "P", "U"]
print(f"样本数: {len(df)}")

# ---------------- 2. 模型计算 ----------------
model = ThermalResistanceModel()
rows = []
for _, s in df.iterrows():
    d = model.thermal_resistances(float(s.w), float(s.r), int(s.n))
    Rstar = model.dimensionless_R(float(s.w), float(s.r), int(s.n), ref="mdotCp")
    rows.append(dict(id=s.id, w=s.w, r=s.r, n=s.n, R_data=s.R,
                     Rstar_model=Rstar, R_total=d["R_total"],
                     R_cond=d["R_cond"], R_conv=d["R_conv"], R_fluid=d["R_fluid"],
                     Re=d["Re"], h_eff=d["h_eff"], A_eff=d["A_eff"]))
md = pd.DataFrame(rows)

# ---------------- 3. 仿射标定（最小二乘） ----------------
X = md["Rstar_model"].values
y = md["R_data"].values
A = np.column_stack([np.ones_like(X), X])
c, *_ = np.linalg.lstsq(A, y, rcond=None)
c0, c1 = c
y_pred = c0 + c1 * X
md["Rstar_calibrated"] = y_pred
md["residual"] = y - y_pred

# ---------------- 4. 验证指标 ----------------
def r2(yy, yp): return 1 - np.sum((yy - yp) ** 2) / np.sum((yy - yy.mean()) ** 2)
def rmse(yy, yp): return float(np.sqrt(np.mean((yy - yp) ** 2)))
def mae(yy, yp):  return float(np.mean(np.abs(yy - yp)))

pearson_raw = np.corrcoef(X, y)[0, 1]
def spearman_rank(x, yy):
        """无 scipy 的 Spearman 秩相关：先求秩再做 Pearson。"""
        def rankdata(a):
            s = pd.Series(a)
            return s.rank(method="average").values
        rx, ry = rankdata(x), rankdata(yy)
        return float(np.corrcoef(rx, ry)[0, 1])
spearman_raw = spearman_rank(X, y)
pearson_cal = np.corrcoef(y_pred, y)[0, 1]

metrics = {
    "样本数": len(df),
    "Pearson(原始模型 vs 数据)": round(float(pearson_raw), 4),
    "Spearman(原始模型 vs 数据)": round(float(spearman_raw), 4),
    "标定系数 c0": round(float(c0), 5),
    "标定系数 c1": round(float(c1), 5),
    "R2(标定后)": round(r2(y, y_pred), 4),
    "RMSE(标定后)": round(rmse(y, y_pred), 5),
    "MAE(标定后)": round(mae(y, y_pred), 5),
    "平均相对误差%(标定后)": round(float(np.mean(np.abs((y - y_pred) / y))) * 100, 2),
    "模型R*范围": f"{X.min():.4f}~{X.max():.4f}",
    "数据R范围": f"{y.min():.4f}~{y.max():.4f}",
}

print("\n================ 验证指标 ================")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# ---------------- 5. 趋势对比表 ----------------
print("\n================ 趋势对比（按参数平均） ================")
print("\n-- 按 w --")
tw = md.groupby("w").agg(R_data=("R_data", "mean"), Rstar=("Rstar_model", "mean"),
                          Rcal=("Rstar_calibrated", "mean")).round(4)
print(tw)
print("\n-- 按 r --")
tr = md.groupby("r").agg(R_data=("R_data", "mean"), Rstar=("Rstar_model", "mean"),
                          Rcal=("Rstar_calibrated", "mean")).round(4)
print(tr)
print("\n-- 按 n --")
tn = md.groupby("n").agg(R_data=("R_data", "mean"), Rstar=("Rstar_model", "mean"),
                          Rcal=("Rstar_calibrated", "mean")).round(4)
print(tn)

# ---------------- 6. 保存 ----------------
md.to_csv(os.path.join(OUT, "thermal_R_model_predictions.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "thermal_R_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("热阻机理模型验证报告\n" + "=" * 40 + "\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("\n趋势对比（按参数平均）\n")
    f.write("\n[按 w]\n"); f.write(tw.to_string())
    f.write("\n\n[按 r]\n"); f.write(tr.to_string())
    f.write("\n\n[按 n]\n"); f.write(tn.to_string())
print(f"\n已保存: {os.path.join(OUT, 'thermal_R_model_predictions.csv')}")
print(f"已保存: {os.path.join(OUT, 'thermal_R_metrics.txt')}")

# ---------------- 7. 绘图 ----------------
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

    # 图1：趋势对比（模型 vs 数据，按 w/r/n 平均）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("w", tw), ("r", tr), ("n", tn)]
    titles = ["针肋宽度比 w", "歧管深高比 r", "针肋排数 n"]
    for ax, (key, tbl), title in zip(axes, panels, titles):
        idx = tbl.index.astype(float) if key != "n" else tbl.index.astype(int)
        ax.plot(idx, tbl["R_data"], "o-", color="crimson", label="附件2数据")
        ax.plot(idx, tbl["Rstar"], "s--", color="steelblue", label="机理模型 R*")
        ax.plot(idx, tbl["Rcal"], "s--", color="darkorange", label="标定后 R*")
        ax.set_xlabel(title); ax.set_ylabel("无量纲热阻")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("热阻模型趋势验证：机理模型 vs 附件2数据（按参数平均）", y=1.02)
    plt.tight_layout()
    f1 = os.path.join(OUT, "thermal_R_trends.png")
    plt.savefig(f1, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f1)

    # 图2：标定前后对比散点（parity）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xx, yy_, ttl) in zip(axes, [
        (X, y, "原始模型（未标定）"), (y_pred, y, "仿射标定后")]):
        ax.scatter(xx, yy_, s=22, alpha=0.6, c="steelblue")
        lo, hi = min(xx.min(), yy_.min()), max(xx.max(), yy_.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        ax.set_xlabel("模型 R*"); ax.set_ylabel("数据 R")
        ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("热阻模型标定前后对比", y=1.02)
    plt.tight_layout()
    f2 = os.path.join(OUT, "thermal_R_parity.png")
    plt.savefig(f2, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f2)

    # 图3：典型方案热阻分解（展示固定底座主导）
    cases = [(0.0, 3.0, 0), (0.1, 3.5, 6), (0.2, 3.5, 4), (0.2, 3.0, 10), (0.2, 4.5, 2), (0.3, 4.5, 4)]
    labels = [f"w={w},r={r},n={n}" for w, r, n in cases]
    Rc = [model.thermal_resistances(w, r, n) for w, r, n in cases]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(cases))
    for key, color, name in [("R_cond", "#9ecae1", "导热 R_cond"),
                             ("R_conv", "#fdae6b", "对流 R_conv"),
                             ("R_fluid", "#a1d99b", "流体升温 R_fluid")]:
        vals = np.array([r[key] for r in Rc])
        ax.bar(labels, vals, bottom=bottom, color=color, label=name)
        bottom += vals
    for i, r in enumerate(Rc):
        ax.text(i, r["R_total"] + 0.002, f"{r['R_total']:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("热阻 (K/W)"); ax.set_title("典型方案热阻分解（R_cond+R_fluid 为结构无关底座）")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    f3 = os.path.join(OUT, "thermal_R_decomposition.png")
    plt.savefig(f3, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f3)
except Exception as e:
    print("绘图失败（不影响结果）:", e)
