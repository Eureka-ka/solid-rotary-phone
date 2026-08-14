# -*- coding: utf-8 -*-
"""
高斯过程回归代理模型 surrogate_gpr.py
======================================
选题B 第二问：用高斯过程回归（GPR / Kriging）建立 (w,r,n) -> (R,P,U) 的代理模型。

- 核函数：RBF（平方指数）+ ARD（每个输入维度独立长度尺度）+ 白噪声
    k(x,x') = sf^2 * exp(-0.5 * sum_d ((x_d-x'_d)/l_d)^2) + sn^2 * delta
- 均值函数：常数（训练时目标标准化为零均值，均值=0）
- 超参数 theta = [log sf, log l_w, log l_r, log l_n, log sn]
    通过最大化对数边际似然估计（无 scipy，用黄金分割坐标下降 + 随机重启）
- 预测：均值 + 方差（不确定性）
- 留一交叉验证 LOOCV：闭式解（用 K^-1 对角），一次求逆即可

作者：Codex（2026-08）
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

def _golden_section(f, a, b, tol: float = 1e-5, max_iter: int = 60) -> float:
    """一维黄金分割最小值搜索"""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
        if abs(b - a) < tol:
            break
    return 0.5 * (a + b)

class GaussianProcessRegressor:
    """RBF-ARD 高斯过程回归（numpy 实现）"""

    # 超参数搜索范围（log 尺度；输入已标准化到[0,1]、目标已标准化）
    BOUNDS = np.array([
        [-3.0, 3.0],   # log sf
        [-4.0, 1.5],   # log l_w
        [-4.0, 1.5],   # log l_r
        [-4.0, 1.5],   # log l_n
        [-6.0, -1.0],  # log sn
    ])

    def __init__(self):
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.K_inv_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.neg_lml_: Optional[float] = None

    # ---------- 核与边际似然 ----------
    def _kernel(self, X1: np.ndarray, X2: np.ndarray, theta: np.ndarray,
                diag_jitter: float = 1e-8) -> np.ndarray:
        """RBF-ARD 核矩阵 K(X1,X2)；对角线加 jitter 保证正定"""
        sf2 = np.exp(2.0 * theta[0])
        ls = np.exp(theta[1:4])
        d = (X1[:, None, :] - X2[None, :, :]) / ls[None, None, :]
        K = sf2 * np.exp(-0.5 * np.sum(d * d, axis=2))
        if X1 is X2 or np.array_equal(X1, X2):
            K = K + diag_jitter * np.eye(len(X1))
        return K

    def _neg_log_marginal_likelihood(self, theta: np.ndarray) -> float:
        K = self._kernel(self.X_, self.X_, theta)
        sn2 = np.exp(2.0 * theta[4])
        Ky = K + sn2 * np.eye(len(self.X_))
        try:
            L = np.linalg.cholesky(Ky)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_))
        n = len(self.y_)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return float(0.5 * self.y_ @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi))

    # ---------- 超参数优化（黄金分割坐标下降 + 随机重启） ----------
    def _optimize_theta(self, n_restarts: int = 10, max_rounds: int = 30) -> Tuple[np.ndarray, float]:
        rng = np.random.default_rng(0)
        best_theta, best_neg = None, 1e18
        for _ in range(n_restarts):
            theta = np.array([rng.uniform(lo, hi) for lo, hi in self.BOUNDS])
            neg = self._neg_log_marginal_likelihood(theta)
            for _round in range(max_rounds):
                changed = False
                for i in range(len(theta)):
                    lo, hi = self.BOUNDS[i]
                    def f(t, i=i, base=theta.copy()):
                        base[i] = t
                        return self._neg_log_marginal_likelihood(base)
                    t_star = _golden_section(f, lo, hi)
                    if abs(t_star - theta[i]) > 1e-4:
                        theta[i] = t_star
                        changed = True
                if not changed:
                    break
            neg = self._neg_log_marginal_likelihood(theta)
            if neg < best_neg:
                best_neg, best_theta = neg, theta.copy()
        return best_theta, best_neg

    # ---------- 训练 ----------
    def fit(self, X: np.ndarray, y: np.ndarray, n_restarts: int = 10,
            max_rounds: int = 30) -> "GaussianProcessRegressor":
        X = np.atleast_2d(X).astype(float)
        y = np.asarray(y, dtype=float).ravel()
        self.y_mean_ = float(y.mean()); self.y_std_ = float(y.std())
        self.X_ = X
        self.y_ = (y - self.y_mean_) / self.y_std_ if self.y_std_ > 0 else y - self.y_mean_
        self.theta_, self.neg_lml_ = self._optimize_theta(n_restarts, max_rounds)
        K = self._kernel(self.X_, self.X_, self.theta_)
        sn2 = np.exp(2.0 * self.theta_[4])
        Ky = K + sn2 * np.eye(len(self.X_))
        L = np.linalg.cholesky(Ky)
        self.K_inv_ = np.linalg.inv(Ky)          # LOOCV 闭式需要
        self.alpha_ = np.linalg.solve(L.T, np.linalg.solve(L, self.y_))
        return self

    # ---------- 预测 ----------
    def predict(self, Xs: np.ndarray, return_std: bool = True):
        Xs = np.atleast_2d(Xs).astype(float)
        Ks = self._kernel(self.X_, Xs, self.theta_)
        mu_std = Ks.T @ self.alpha_
        mu = self.y_mean_ + self.y_std_ * mu_std
        if not return_std:
            return mu
        kss = self._kernel(Xs, Xs, self.theta_)
        sn2 = np.exp(2.0 * self.theta_[4])
        var_std = np.maximum(np.diag(kss) - np.sum(Ks.T * (self.K_inv_ @ Ks).T, axis=1) + sn2, 0.0)
        var = self.y_std_ ** 2 * var_std
        return mu, np.sqrt(var)

    # ---------- 闭式 LOOCV ----------
    def loocv(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (预测均值, 预测标准差)，均为原始尺度"""
        Kinv_y = self.K_inv_ @ self.y_
        Kinv_diag = np.diag(self.K_inv_)
        mu_loocv_std = self.y_ - Kinv_y / Kinv_diag
        var_loocv_std = 1.0 / Kinv_diag
        mu = self.y_mean_ + self.y_std_ * mu_loocv_std
        sd = self.y_std_ * np.sqrt(np.maximum(var_loocv_std, 0.0))
        return mu, sd

    @property
    def hyperparameters(self) -> dict:
        if self.theta_ is None:
            return {}
        return dict(sf=float(np.exp(self.theta_[0])),
                    l_w=float(np.exp(self.theta_[1])),
                    l_r=float(np.exp(self.theta_[2])),
                    l_n=float(np.exp(self.theta_[3])),
                    sn=float(np.exp(self.theta_[4])))

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(40, 3))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2]
    gp = GaussianProcessRegressor().fit(X, y, n_restarts=6)
    mu, sd = gp.predict(X[:5])
    print("超参数:", gp.hyperparameters)
    print("预测:", mu[:3], "±", sd[:3])
    loocv_mu, loocv_sd = gp.loocv()
    r2 = 1 - np.sum((y - loocv_mu) ** 2) / np.sum((y - y.mean()) ** 2)
    print("LOOCV R2 (自检):", round(r2, 4))
