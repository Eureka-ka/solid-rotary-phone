# -*- coding: utf-8 -*-
"""
run_q5.py — Q5 加工误差与工况波动的敏感性分析
================================================
不确定参数：
  加工误差：w ±5%（均匀）、r ±5%（均匀，截断到[3,4.5]）、n ±1 排（离散，取{0,2,4,6,8,10}最近值）
  工况波动：ṁ ~ N(1 g/s, 0.05²)、T_in ~ N(293 K, 1²)、热源功率 q ±10%（均匀）
性能计算（分层模型）：
  w/r/n → Q2 高斯过程（数据精确代理）
  ṁ/T_in → Q1 机理模型缩放因子（GP 无工况输入，用机理相对变化补充；q 对无量纲指标无影响）
方法：局部敏感性(LSA 弹性系数) + 全局敏感性(Sobol' 一阶/总效应) + 蒙特卡洛(N=10000)
     + 稳定性指标(CV/SNR/可靠性β/失效概率 P(>1.10×标称)) + 方案对比
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from surrogate_gpr import GaussianProcessRegressor
from thermal_resistance_model import ThermalResistanceModel, Geometry, WATER, CONDITIONS
from pressure_drop_model import PressureDropModel
from nonuniformity_model import NonuniformityModel

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

# ---------------- 1. 训练 GP ----------------
df = pd.read_excel(DATA_XLSX, header=1)
df.columns = ["id", "w", "r", "n", "R", "P", "U"]
Xr = df[["w", "r", "n"]].values.astype(float)
xmin, xmax = Xr.min(0), Xr.max(0)
def norm(X): return (np.atleast_2d(X) - xmin) / (xmax - xmin)
gps = {}
for t in ["R", "P", "U"]:
    gps[t] = GaussianProcessRegressor().fit(norm(Xr), df[t].values, n_restarts=8)
    print(f"GP {t} 就绪")

# ---------------- 2. 机理缩放因子（ṁ × T_in 网格） ----------------
U_COEF = [0.7427, 0.1012, 1.524, 0.0002, 0.0045, 0.015]   # 非均匀性标定系数
D0 = (0.225, 4.5, 4)
def mu_T(T):  # 水动力黏度随温度近似（293K 附近约 -2.25%/K）
    return WATER["mu"] * (1.0 - 0.0225 * (T - 293.0))

m_axis = np.linspace(0.9, 1.1, 21)      # g/s
T_axis = np.linspace(288, 298, 21)      # K
print("计算机理缩放因子网格 ...")
F_R = np.zeros((len(m_axis), len(T_axis)))
F_P = np.zeros_like(F_R); F_U = np.zeros_like(F_R)
for i, m in enumerate(m_axis):
    for j, T in enumerate(T_axis):
        cond = dict(CONDITIONS, mdot=m * 1e-3)
        water = dict(WATER, mu=mu_T(T))
        Rm = ThermalResistanceModel(water=water, conditions=cond).thermal_resistances(*D0)["R_total"]
        Pm = PressureDropModel(water=water, conditions=cond).components(*D0)["dP_total"]
        Um = NonuniformityModel(water=water, conditions=cond).predict(*D0, U_COEF)
        F_R[i, j], F_P[i, j], F_U[i, j] = Rm, Pm, Um
R0m, P0m, U0m = F_R[10, 10], F_P[10, 10], F_U[10, 10]     # (ṁ=1,T=293)
F_R /= R0m; F_P /= P0m; F_U /= U0m
print(f"  机理标称: R={R0m:.4f}, P={P0m:.0f} Pa, U={U0m:.4f}")

def bilinear(F, m, T):
    m = np.clip(np.asarray(m, float), m_axis[0], m_axis[-1])
    T = np.clip(np.asarray(T, float), T_axis[0], T_axis[-1])
    im = np.clip(np.searchsorted(m_axis, m) - 1, 0, len(m_axis) - 2)
    iT = np.clip(np.searchsorted(T_axis, T) - 1, 0, len(T_axis) - 2)
    wm = (m - m_axis[im]) / (m_axis[im + 1] - m_axis[im])
    wT = (T - T_axis[iT]) / (T_axis[iT + 1] - T_axis[iT])
    return (F[im, iT] * (1 - wm) * (1 - wT) + F[im + 1, iT] * wm * (1 - wT)
            + F[im, iT + 1] * (1 - wm) * wT + F[im + 1, iT + 1] * wm * wT)

# ---------------- 3. 分层组合模型 ----------------
def _gp_batch(gp, Rn, chunk=1500):
    """分块 GP 预测（避免大样本时自核矩阵 kss 内存爆炸）"""
    outs = []
    for i in range(0, len(Rn), chunk):
        mu, _ = gp.predict(Rn[i:i + chunk])
        outs.append(np.asarray(mu))
    return np.concatenate(outs)

def combined_batch(X):
    """X: (N,6)=[w,r,n,m(g/s),T(K),q] → (R,P,U) 三列"""
    X = np.atleast_2d(np.asarray(X, float))
    Rn = norm(X[:, :3])
    R0 = _gp_batch(gps["R"], Rn)
    P0 = _gp_batch(gps["P"], Rn)
    U0 = _gp_batch(gps["U"], Rn)
    fR = bilinear(F_R, X[:, 3], X[:, 4])
    fP = bilinear(F_P, X[:, 3], X[:, 4])
    fU = bilinear(F_U, X[:, 3], X[:, 4])
    return np.column_stack([R0 * fR, P0 * fP, U0 * fU])

# ---------------- 4. 局部敏感性（弹性系数，+1% 有限差分） ----------------
print("\n=== 局部敏感性（弹性系数 S=(ΔY/Y)/(Δx/x)，标称 (0.225,4.5,4)）===")
params = ["w", "r", "n", "ṁ", "T_in", "q"]
nom = np.array([0.225, 4.5, 4, 1.0, 293.0, 5e9])
Y0 = combined_batch(nom[None, :])[0]
bounds = np.array([(0.1, 0.3), (3.0, 4.5), (0.0, 10.0), (0.9, 1.1), (288.0, 298.0), (0.9e10, 1.1e10)])
LSA = np.zeros((len(params), 3))
for k in range(len(params)):
    x_hi = np.clip(nom[k] * 1.01, *bounds[k])
    x_lo = np.clip(nom[k] * 0.99, *bounds[k])
    xp = nom.copy()
    xp[k] = x_hi if abs(x_hi - nom[k]) > 1e-12 else x_lo   # 边界参数取向内方向扰动
    Yp = combined_batch(xp[None, :])[0]
    LSA[k] = ((Yp - Y0) / Y0) / ((xp[k] - nom[k]) / nom[k])
lsa_df = pd.DataFrame(LSA, index=params, columns=["R", "P", "U"]).round(3)
print(lsa_df.to_string())
lsa_df.to_csv(unlocked_path(os.path.join(OUT, "q5_lsa_elasticity.csv")), encoding="utf-8-sig")

# ---------------- 5. Sobol' 全局敏感性（Saltelli） ----------------
def erfinv(x):
    a = 0.147
    ln = np.log(1 - x * x + 1e-300)
    t1 = 2.0 / (np.pi * a) + ln / 2.0
    t2 = ln / a
    return np.sign(x) * np.sqrt(np.sqrt(t1 * t1 - t2) - t1)

def sample_uniform(N, rng):
    return rng.random((N, 6))   # 6 个参数，先均匀[0,1]

def transform(U):
    """均匀[0,1] → 实际分布（Sobol' 用，保持连续、无点质量）。
    列顺序=[w,r,n,ṁ,T,q]。r=4.5 在边界→只取向内范围 U(4.275,4.5)；n 连续[3,5]。"""
    X = np.empty_like(U)
    X[:, 0] = 0.225 * (1 + 0.05 * (2 * U[:, 0] - 1))                 # w ±5% 均匀
    X[:, 1] = 4.275 + 0.225 * U[:, 1]                                # r ~ U(4.275,4.5)（边界向内）
    X[:, 2] = 3.0 + 2.0 * U[:, 2]                                    # n 连续 [3,5]
    X[:, 3] = 1.0 + 0.05 * erfinv(2 * U[:, 3] - 1)                   # ṁ ~ N(1,0.05²)
    X[:, 4] = 293.0 + 1.0 * erfinv(2 * U[:, 4] - 1)                  # T ~ N(293,1²)
    X[:, 5] = 5e9 * (1 + 0.1 * (2 * U[:, 5] - 1))                    # q ±10%
    return X

print("\n=== 全局敏感性（总效应 ST=Saltelli 稳健估计；一阶 S1=暴力相关比定义式）===")
# ---- ST（总效应）：Saltelli，N=10000 ----
rng = np.random.default_rng(0)
N_ST = 10000
A = transform(rng.random((N_ST, 6)))
B = transform(rng.random((N_ST, 6)))
rows = [A, B]
for k in range(6):
    AB = A.copy(); AB[:, k] = B[:, k]; rows.append(AB)
    BA = B.copy(); BA[:, k] = A[:, k]; rows.append(BA)
Yall = combined_batch(np.vstack(rows))
n = N_ST
YA, YB = Yall[:n], Yall[n:2 * n]
f0sq = np.mean(YA * YB, axis=0)
VarY = (np.mean(YA ** 2, axis=0) + np.mean(YB ** 2, axis=0)) / 2 - f0sq
ST = np.zeros((6, 3))
for k in range(6):
    AB = Yall[2 * n + 2 * k * n: 2 * n + (2 * k + 1) * n]
    ST[k] = np.mean((YA - AB) ** 2, axis=0) / (2 * VarY)

# ---- S1（一阶）：暴力相关比 Var(E[Y|X_i])/Var(Y)，N=200000 ----
N_S1 = 200000
U1 = rng.random((N_S1, 6))
X1 = transform(U1)
Y1 = combined_batch(X1)
Var1 = Y1.var(axis=0)
K = 60
S1 = np.zeros((6, 3))
for k in range(6):
    v = X1[:, k]
    edges = np.linspace(v.min(), v.max(), K + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, K - 1)
    Ey = np.zeros((K, 3)); cnt = np.zeros(K)
    for b in range(K):
        msk = idx == b
        cnt[b] = msk.sum()
        if cnt[b] > 0:
            Ey[b] = Y1[msk].mean(axis=0)
    wgt = cnt / cnt.sum()
    mean_ey = np.sum(wgt[:, None] * Ey, axis=0)
    S1[k] = np.sum(wgt[:, None] * (Ey - mean_ey) ** 2, axis=0) / Var1

sobol = pd.DataFrame(np.hstack([S1, ST]), index=params,
                     columns=["S1_R", "S1_P", "S1_U", "ST_R", "ST_P", "ST_U"]).round(3)
print(sobol.to_string())
sobol.to_csv(unlocked_path(os.path.join(OUT, "q5_sobol_indices.csv")), encoding="utf-8-sig")
print("已保存 q5_lsa_elasticity.csv / q5_sobol_indices.csv")


# ---------------- 6. 蒙特卡洛仿真（鲁棒方案 + 方案对比） ----------------
print("\n=== 蒙特卡洛仿真（N=10000，真实离散化：r截断、n取最近有效排数）===")
def mc_sample(design, N=10000, seed=1):
    rng = np.random.default_rng(seed)
    w0, r0, n0 = design
    w = w0 * (1 + 0.05 * (2 * rng.random(N) - 1))                    # ±5% 均匀
    w = np.clip(w, 0.1, 0.3)
    r = r0 * (1 + 0.05 * (2 * rng.random(N) - 1))                    # ±5%，截断[3,4.5]
    r = np.clip(r, 3.0, 4.5)
    n = np.round(n0 + (2 * rng.random(N) - 1)).astype(int)           # ±1 排
    nvals = np.array([0, 2, 4, 6, 8, 10])
    n = nvals[np.argmin(np.abs(n[:, None] - nvals[None, :]), axis=1)]
    m = rng.normal(1.0, 0.05, N)                                     # N(1,0.05²)
    T = rng.normal(293.0, 1.0, N)                                    # N(293,1²)
    q = 5e9 * (1 + 0.1 * (2 * rng.random(N) - 1))                    # ±10%
    return np.column_stack([w, r, n, m, T, q])

schemes = {
    "Q3综合最优(0.215,4.5,4)": (0.215, 4.5, 4),
    "Q4鲁棒(0.225,4.5,4)": (0.225, 4.5, 4),
    "低R(0.2,3,10)": (0.2, 3.0, 10),
    "低P(0.2,4.5,2)": (0.2, 4.5, 2),
}
summary = []
mc_samples = {}
for name, d0 in schemes.items():
    Xmc = mc_sample(d0)
    Y = combined_batch(Xmc)
    Y0 = combined_batch(np.array([[d0[0], d0[1], d0[2], 1.0, 293.0, 5e9]]))[0]
    thr = 1.10 * Y0
    for k, t in enumerate(["R", "P", "U"]):
        y = Y[:, k]
        fail = float(np.mean(y > thr[k]))
        cv = float(y.std() / y.mean())
        snr = float(-10 * np.log10(np.mean(y ** 2)))     # 田口望小 SNR
        beta = float((thr[k] - y.mean()) / y.std())
        summary.append(dict(方案=name, 指标=t, 标称=round(Y0[k], 5),
                            mean=round(float(y.mean()), 5), std=round(float(y.std()), 5),
                            CV=round(cv, 4), P5=round(float(np.percentile(y, 5)), 5),
                            P95=round(float(np.percentile(y, 95)), 5),
                            SNR=round(snr, 3), beta=round(beta, 3),
                            失效概率=round(fail, 5)))
    mc_samples[name] = Y
    print(f"  {name}: R={Y[:,0].mean():.4f}±{Y[:,0].std():.4f} P={Y[:,1].mean():.4f}±{Y[:,1].std():.4f} "
          f"U={Y[:,2].mean():.4f}±{Y[:,2].std():.4f}")

ms = pd.DataFrame(summary)
ms.to_csv(unlocked_path(os.path.join(OUT, "q5_mc_summary.csv")), index=False, encoding="utf-8-sig")
print(ms.to_string(index=False))
print("\n已保存 q5_mc_summary.csv")

# 保存鲁棒方案 MC 性能样本（供绘图）
rob = pd.DataFrame(mc_samples["Q4鲁棒(0.225,4.5,4)"], columns=["R", "P", "U"])
rob.to_csv(unlocked_path(os.path.join(OUT, "q5_mc_samples_robust.csv")), index=False, encoding="utf-8-sig")


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
    plab = ["w", "r", "n", "流量", "入口温度", "q"]   # 显示标签（避免 ṁ 字形缺失）

    # 图1：Sobol 柱状图（每输出一子图，S1 与 ST）
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(params))
    for ax, t in zip(axes, ["R", "P", "U"]):
        s1 = sobol["S1_" + t].values; st = sobol["ST_" + t].values
        ax.bar(x - 0.2, s1, width=0.4, label="S1 一阶", color="steelblue")
        ax.bar(x + 0.2, st, width=0.4, label="ST 总效应", color="#fdae6b")
        ax.set_xticks(x); ax.set_xticklabels(plab, fontsize=9)
        ax.set_title(f"{t} 的 Sobol 指数"); ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Q5 全局敏感性：Sobol 一阶与总效应指数", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q5_sobol.png"), dpi=140, bbox_inches="tight"); plt.close()
    print("已保存: q5_sobol.png")

    # 图2：鲁棒方案 MC 概率密度
    Yr = mc_samples["Q4鲁棒(0.225,4.5,4)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, t, k in zip(axes, ["R", "P", "U"], range(3)):
        y = Yr[:, k]
        ax.hist(y, bins=60, color="steelblue", alpha=0.8, density=True)
        ax.axvline(np.percentile(y, 5), color="tab:red", ls="--", lw=1, label="P5")
        ax.axvline(np.percentile(y, 95), color="tab:red", ls="--", lw=1, label="P95")
        ax.axvline(y.mean(), color="k", lw=1.2, label="均值")
        ax.set_xlabel(t); ax.set_ylabel("概率密度"); ax.set_title(f"{t} 扰动分布 (CV={y.std()/y.mean():.3f})")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Q5 蒙特卡洛：鲁棒方案 (0.225,4.5,4) 的 R/P/U 扰动分布（N=10000）", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q5_mc_pdf.png"), dpi=140, bbox_inches="tight"); plt.close()
    print("已保存: q5_mc_pdf.png")

    # 图3：Tornado（弹性系数）
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, t in zip(axes, ["R", "P", "U"]):
        vals = lsa_df[t].values
        order = np.argsort(np.abs(vals))
        ax.barh(np.arange(len(params)), vals[order], color=["#fdae6b" if v >= 0 else "steelblue" for v in vals[order]])
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels([plab[i] for i in order], fontsize=9)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{t} 弹性系数"); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Q5 局部敏感性：归一化弹性系数（Tornado）", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q5_tornado.png"), dpi=140, bbox_inches="tight"); plt.close()
    print("已保存: q5_tornado.png")

    # 图4：二维扰动热力图（w × ṁ，标称 r=4.5,n=4）
    wg = np.linspace(0.225*0.95, 0.225*1.05, 60)
    mg = np.linspace(0.9, 1.1, 60)
    Wg, Mg = np.meshgrid(wg, mg)
    Xg = np.column_stack([Wg.ravel(), np.full(Wg.size, 4.5), np.full(Wg.size, 4),
                          Mg.ravel(), np.full(Wg.size, 293.0), np.full(Wg.size, 5e9)])
    Yg = combined_batch(Xg)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, t, k in zip(axes, ["R", "P", "U"], range(3)):
        im = ax.pcolormesh(Wg, Mg, Yg[:, k].reshape(Wg.shape), cmap="viridis", shading="auto")
        ax.set_xlabel("w（±5% 扰动）"); ax.set_ylabel("流量 (g/s)（±10% 扰动）")
        ax.set_title(f"{t} 响应"); fig.colorbar(im, ax=ax)
    fig.suptitle("Q5 二维参数扰动热力图（标称 r=4.5, n=4）", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q5_2d_heatmap.png"), dpi=140, bbox_inches="tight"); plt.close()
    print("已保存: q5_2d_heatmap.png")
except Exception as e:
    print("绘图失败:", e)
