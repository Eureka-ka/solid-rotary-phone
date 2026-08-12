# -*- coding: utf-8 -*-
"""
run_pressure_drop.py — 压降模型主程序
1) 加载附件2数据（P 无量纲压降）
2) 机理模型计算四项压降分量（dP_f/dP_pin/dP_man/dP_turn，Pa）
3) 最小二乘拟合 P*_pred = c0 + c_f·dP_f + c_pin·dP_pin + c_man·dP_man + c_turn·dP_turn
   （分量权重由数据标定，兼顾量纲归一化与相对贡献）
4) 验证：Pearson/Spearman、R²、RMSE、MAE、平均相对误差、趋势对比表
5) 输出 CSV、指标文本、三张图
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from pressure_drop_model import PressureDropModel

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
DATA_XLSX = glob.glob(r"E:\数学建模大赛\选题D\选题B\附件\*.xlsx")[0]

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

m = PressureDropModel()
rows = []
for _, s in df.iterrows():
    c = m.components(float(s.w), float(s.r), int(s.n))
    rows.append(dict(id=s.id, P_data=s.P, **c))
md = pd.DataFrame(rows)

# 最小二乘：P* = c0 + c_f*dP_f + c_pin*dP_pin + c_man*dP_man + c_turn*dP_turn
feat = ["dP_f", "dP_pin", "dP_man", "dP_turn"]
X = np.column_stack([np.ones(len(md))] + [md[f].values for f in feat])
y = md["P_data"].values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ coef
md["P_calibrated"] = y_pred
md["residual"] = y - y_pred

c0 = coef[0]
coefs = dict(zip(feat, coef[1:]))

def r2(yy, yp): return 1 - np.sum((yy - yp) ** 2) / np.sum((yy - yy.mean()) ** 2)
def spearman(x, yy):
    def rank(a):
        return pd.Series(a).rank(method="average").values
    return float(np.corrcoef(rank(x), rank(yy))[0, 1])

metrics = {
    "样本数": len(df),
    "Pearson(原始dP_total vs 数据)": round(float(np.corrcoef(md.dP_total, y)[0, 1]), 4),
    "Spearman(原始dP_total vs 数据)": round(spearman(md.dP_total.values, y), 4),
    "截距 c0": round(float(c0), 5),
    "c_f(沿程)": round(float(coefs["dP_f"]), 6),
    "c_pin(针肋)": round(float(coefs["dP_pin"]), 6),
    "c_man(歧管)": round(float(coefs["dP_man"]), 6),
    "c_turn(转弯)": round(float(coefs["dP_turn"]), 6),
    "R2(标定后)": round(r2(y, y_pred), 4),
    "RMSE": round(float(np.sqrt(np.mean((y - y_pred) ** 2))), 5),
    "MAE": round(float(np.mean(np.abs(y - y_pred))), 5),
    "平均相对误差%": round(float(np.mean(np.abs((y - y_pred) / y))) * 100, 2),
    "模型dP_total范围(Pa)": f"{md.dP_total.min():.0f}~{md.dP_total.max():.0f}",
    "数据P范围": f"{y.min():.4f}~{y.max():.4f}",
}

print("\n================ 验证指标 ================")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# 各分量相对贡献（按标定权重加权后的均值占比）
w_contrib = {f: abs(coefs[f]) * md[f].mean() for f in feat}
tot = sum(w_contrib.values())
print("\n标定后各分量对无量纲 P* 的平均贡献占比:")
for f in feat:
    print(f"  {f}: {w_contrib[f]/tot*100:.1f}%")

print("\n================ 趋势对比（按参数平均） ================")
for key, name in [("w", "w"), ("r", "r"), ("n", "n")]:
    tbl = md.groupby(key).agg(P_data=("P_data", "mean"), dP=("dP_total", "mean"),
                              Pcal=("P_calibrated", "mean")).round(4)
    print(f"\n-- 按 {name} --"); print(tbl)

# 保存
csv_path = unlocked_path(os.path.join(OUT, "pressure_drop_model_predictions.csv"))
md.to_csv(csv_path, index=False, encoding="utf-8-sig")
txt_path = unlocked_path(os.path.join(OUT, "pressure_drop_metrics.txt"))
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("压降机理模型验证报告\n" + "=" * 40 + "\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("\n标定后分量贡献占比\n")
    for feat_name in feat:
        f.write(f"{feat_name}: {w_contrib[feat_name]/tot*100:.1f}%\n")
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title in zip(axes, ["w", "r", "n"], ["针肋宽度比 w", "歧管深高比 r", "针肋排数 n"]):
        tbl = md.groupby(key).agg(P_data=("P_data", "mean"), Pcal=("P_calibrated", "mean"))
        idx = tbl.index.astype(float) if key != "n" else tbl.index.astype(int)
        ax.plot(idx, tbl["P_data"], "o-", color="crimson", label="附件2数据")
        ax.plot(idx, tbl["Pcal"], "s--", color="steelblue", label="模型（标定后）")
        ax.set_xlabel(title); ax.set_ylabel("无量纲压降 P*")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("压降模型趋势验证：机理模型 vs 附件2数据（按参数平均）", y=1.02)
    plt.tight_layout()
    f1 = os.path.join(OUT, "pressure_drop_trends.png")
    plt.savefig(f1, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(md.dP_total, y, s=22, alpha=0.6, c="steelblue")
    axes[0].set_xlabel("模型 ΔP_total (Pa)"); axes[0].set_ylabel("数据 P*")
    axes[0].set_title("原始 ΔP_total vs 数据"); axes[0].grid(alpha=0.3)
    axes[1].scatter(y_pred, y, s=22, alpha=0.6, c="tab:green")
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].set_xlabel("模型 P*（标定后）"); axes[1].set_ylabel("数据 P*")
    axes[1].set_title("标定后对比"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    f2 = os.path.join(OUT, "pressure_drop_parity.png")
    plt.savefig(f2, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f2)

    # 分量分解（代表性方案）
    cases = [(0.0, 3.0, 0), (0.2, 3.0, 10), (0.2, 3.5, 4), (0.2, 4.5, 2), (0.3, 4.5, 4)]
    labels = [f"w={w},r={r},n={n}" for w, r, n in cases]
    comps = [m.components(w, r, n) for w, r, n in cases]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(cases))
    for key, color, name in [("dP_f", "#9ecae1", "沿程 ΔP_f"),
                             ("dP_pin", "#fdae6b", "针肋 ΔP_pin"),
                             ("dP_man", "#a1d99b", "歧管 ΔP_man"),
                             ("dP_turn", "#bcbddc", "转弯 ΔP_turn")]:
        vals = np.array([c[key] for c in comps])
        ax.bar(labels, vals, bottom=bottom, color=color, label=name)
        bottom += vals
    for i, c in enumerate(comps):
        ax.text(i, c["dP_total"] + 150, f"{c['dP_total']:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("ΔP (Pa)"); ax.set_title("典型方案压降分量分解")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    f3 = os.path.join(OUT, "pressure_drop_decomposition.png")
    plt.savefig(f3, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f3)
except Exception as e:
    print("绘图失败（不影响结果）:", e)
