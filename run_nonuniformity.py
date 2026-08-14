# -*- coding: utf-8 -*-
"""
run_nonuniformity.py — 温度非均匀性模型主程序
1) 加载附件2 U 数据
2) 机理特征：x_mal（分配不均匀数）、x_w、x_n（局部强化非均匀）
3) 最小二乘标定 U* = c0 + c1*x_mal + c2*x_w + c3*x_n
4) 验证：Pearson/Spearman、R²、RMSE、MAE、平均相对误差、趋势表
5) 输出 CSV、指标文本、三张图
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from nonuniformity_model import NonuniformityModel

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

m = NonuniformityModel()
rows = []
for _, s in df.iterrows():
    f = m.features(float(s.w), float(s.r), int(s.n))
    rows.append(dict(id=s.id, U_data=s.U, **f))
md = pd.DataFrame(rows)

feat = ["x_mal", "x_w", "x_n", "n_lin", "x_r2"]
X = np.column_stack([np.ones(len(md))] + [md[g].values for g in feat])
y = md["U_data"].values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
y_pred = X @ coef
md["U_calibrated"] = y_pred
md["residual"] = y - y_pred

def r2(yy, yp): return 1 - np.sum((yy - yp) ** 2) / np.sum((yy - yy.mean()) ** 2)
def spearman(x, yy):
    def rank(a):
        return pd.Series(a).rank(method="average").values
    return float(np.corrcoef(rank(x), rank(yy))[0, 1])

metrics = {
    "样本数": len(df),
    "Pearson(原始特征线性组合 vs 数据)": round(float(np.corrcoef(md.x_mal, y)[0, 1]), 4),
    "Spearman": round(spearman(md.x_mal.values, y), 4),
    "c0(截距)": round(float(coef[0]), 4),
    "c1(x_mal 分配不均匀)": round(float(coef[1]), 4),
    "c2(x_w 局部w扰动)": round(float(coef[2]), 4),
    "c3(x_n 局部n扰动)": round(float(coef[3]), 4),
    "c4(n 线性扰动)": round(float(coef[4]), 4),
    "c5((r-r_opt)^2)": round(float(coef[5]), 4),
    "R2(标定后)": round(r2(y, y_pred), 4),
    "RMSE": round(float(np.sqrt(np.mean((y - y_pred) ** 2))), 5),
    "MAE": round(float(np.mean(np.abs(y - y_pred))), 5),
    "平均相对误差%": round(float(np.mean(np.abs((y - y_pred) / y))) * 100, 2),
    "模型U*范围": f"{y_pred.min():.4f}~{y_pred.max():.4f}",
    "数据U范围": f"{y.min():.4f}~{y.max():.4f}",
}
print("\n================ 验证指标 ================")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print("\n================ 趋势对比（按参数平均） ================")
for key in ["w", "r", "n"]:
    tbl = md.groupby(key).agg(U_data=("U_data", "mean"), Ucal=("U_calibrated", "mean")).round(4)
    print(f"\n-- 按 {key} --"); print(tbl)

csv_path = unlocked_path(os.path.join(OUT, "nonuniformity_model_predictions.csv"))
md.to_csv(csv_path, index=False, encoding="utf-8-sig")
txt_path = unlocked_path(os.path.join(OUT, "nonuniformity_metrics.txt"))
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("温度非均匀性机理模型验证报告\n" + "=" * 40 + "\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
print(f"\n已保存: {csv_path}")
print(f"已保存: {txt_path}")

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
        tbl = md.groupby(key).agg(U_data=("U_data", "mean"), Ucal=("U_calibrated", "mean"))
        idx = tbl.index.astype(float) if key != "n" else tbl.index.astype(int)
        ax.plot(idx, tbl["U_data"], "o-", color="crimson", label="附件2数据")
        ax.plot(idx, tbl["Ucal"], "s--", color="steelblue", label="模型（标定后）")
        ax.set_xlabel(title); ax.set_ylabel("无量纲温度非均匀性 U*")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("温度非均匀性模型趋势验证：机理模型 vs 附件2数据", y=1.02)
    plt.tight_layout()
    f1 = os.path.join(OUT, "nonuniformity_trends.png")
    plt.savefig(f1, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(md.x_mal, y, s=22, alpha=0.6, c="steelblue")
    axes[0].set_xlabel("x_mal（分配不均匀数）"); axes[0].set_ylabel("数据 U*")
    axes[0].set_title("分配不均匀数 vs 数据 U*"); axes[0].grid(alpha=0.3)
    axes[1].scatter(y_pred, y, s=22, alpha=0.6, c="tab:green")
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].set_xlabel("模型 U*（标定后）"); axes[1].set_ylabel("数据 U*")
    axes[1].set_title("标定后对比"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    f2 = os.path.join(OUT, "nonuniformity_parity.png")
    plt.savefig(f2, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f2)

    # 特征贡献
    contrib = {g: abs(coef[i + 1]) * md[g].mean() for i, g in enumerate(feat)}
    tot = sum(contrib.values())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(contrib.keys()), [v / tot * 100 for v in contrib.values()],
           color=["#9ecae1", "#fdae6b", "#a1d99b"])
    for i, (g, v) in enumerate(contrib.items()):
        ax.text(i, v / tot * 100 + 2, f"{v/tot*100:.1f}%", ha="center")
    ax.set_ylabel("贡献占比 (%)"); ax.set_title("各机理特征对 U* 的标定贡献占比")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    f3 = os.path.join(OUT, "nonuniformity_contrib.png")
    plt.savefig(f3, dpi=140, bbox_inches="tight"); plt.close()
    print("已保存:", f3)
except Exception as e:
    print("绘图失败（不影响结果）:", e)
