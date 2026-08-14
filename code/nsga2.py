# -*- coding: utf-8 -*-
"""
多目标优化工具 nsga2.py
========================
- 非支配排序（分块向量化，用于网格穷举 Pareto 前沿）
- 快速非支配排序 + 拥挤度（NSGA-II 用）
- NSGA-II（实数编码 + SBX + 多项式变异，numpy 实现，无外部依赖）

目标均为最小化。设计变量 x = [w, r, n]，n 取离散值 {0,2,4,6,8,10}；
物理约束：w=0 ⇒ n=0（评价时修复）。
"""
from __future__ import annotations
import numpy as np

DISCRETE_N = np.array([0, 2, 4, 6, 8, 10], dtype=float)

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a 支配 b（最小化）：a 各分量<=b 且至少一个严格<"""
    return bool(np.all(a <= b) and np.any(a < b))

def pareto_front_indices(values: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """对候选集 values (n,m) 返回 Pareto 前沿（非支配点）的下标（分块向量化）。
    点 i 被移除当且仅当存在 j 使 values[j] 支配 values[i]（<= 且至少一个 <）。"""
    n = len(values)
    keep = np.ones(n, dtype=bool)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        vc = values[start:end]
        # values[j] 支配 vc[i] ?
        le = np.all(values[None, :, :] <= vc[:, None, :], axis=2)   # (k,n): values[j]<=vc[i]
        lt = np.any(values[None, :, :] < vc[:, None, :], axis=2)
        dom = le & lt
        keep[start:end] = ~dom.any(axis=1)
    return np.where(keep)[0]

def fast_non_dominated_sort(values: np.ndarray):
    """返回分层前沿列表 [[idx...], ...]（O(N^2)，适合小种群）"""
    n = len(values)
    S = [[] for _ in range(n)]
    n_p = np.zeros(n, dtype=int)
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(values[i], values[j]):
                S[i].append(j)
            elif dominates(values[j], values[i]):
                n_p[i] += 1
        if n_p[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        Q = []
        for i in fronts[k]:
            for j in S[i]:
                n_p[j] -= 1
                if n_p[j] == 0:
                    Q.append(j)
        k += 1
        fronts.append(Q)
    return fronts[:-1]

def crowding_distance(values: np.ndarray) -> np.ndarray:
    """同一前沿内各点的拥挤度（越大越分散）"""
    n = len(values)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    m = values.shape[1]
    for j in range(m):
        order = np.argsort(values[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        rng = values[order[-1], j] - values[order[0], j]
        if rng < 1e-12:
            continue
        for t in range(1, n - 1):
            dist[order[t]] += (values[order[t + 1], j] - values[order[t - 1], j]) / rng
    return dist

def _sbx(parent1: np.ndarray, parent2: np.ndarray, eta_c: float, rng) -> np.ndarray:
    """模拟二进制交叉 SBX（实数连续基因）"""
    u = rng.random(len(parent1))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                    1 / (2 * (1 - u)) ** (1 / (eta_c + 1)))
    child = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    return child

def _polynomial_mutation(child: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                         pm: float, eta_m: float, rng) -> np.ndarray:
    """多项式变异（连续基因）"""
    delta = np.where(rng.random(len(child)) < pm, 1.0, 0.0)
    u = rng.random(len(child))
    r = np.where(u < 0.5,
                 (2 * u) ** (1 / (eta_m + 1)) - 1,
                 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    child = child + delta * r * (hi - lo)
    return child

class NSGA2:
    """实数编码 NSGA-II（最小化 3 目标）。基因 x=[w,r,n]，n 评价时取离散值。"""

    def __init__(self, bounds: list = [(0.0, 0.3), (3.0, 4.5), (0.0, 10.0)],
                 pop_size: int = 120, n_gen: int = 300, pc: float = 0.9,
                 pm: float = 0.4, eta_c: float = 15, eta_m: float = 20, seed: int = 42):
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.pc = pc
        self.pm = pm
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.rng = np.random.default_rng(seed)

    def _repair(self, X: np.ndarray) -> np.ndarray:
        """边界修复 + n 离散化 + w=0⇒n=0"""
        X = np.clip(X, self.bounds[:, 0], self.bounds[:, 1])
        X = X.copy()
        X[:, 2] = DISCRETE_N[np.argmin(np.abs(X[:, 2][:, None] - DISCRETE_N[None, :]), axis=1)]
        # w=0（或极接近0）⇒ n=0
        near_zero = X[:, 0] < 0.005
        X[near_zero, 2] = 0.0
        X[near_zero, 0] = 0.0
        # n=0 但 w>0（无物理意义）→ 修复为最小正排数 n=2
        X[(X[:, 2] == 0) & (X[:, 0] >= 0.005), 2] = 2.0
        return X

    def run(self, objective_fn):
        """objective_fn(X(n,3)) -> F(n,m) 最小化。返回 (前沿X, 前沿F)"""
        lo = self.bounds[:, 0]; hi = self.bounds[:, 1]
        X = self.rng.uniform(lo, hi, size=(self.pop_size, 3))
        X = self._repair(X)
        F = np.asarray(objective_fn(X))
        for _gen in range(self.n_gen):
            # 锦标赛选择（rank + 拥挤度）
            fronts = fast_non_dominated_sort(F)
            rank = np.empty(self.pop_size, dtype=int)
            cd = np.zeros(self.pop_size)
            for r, fr in enumerate(fronts):
                for i in fr:
                    rank[i] = r
                cd_arr = crowding_distance(F[fr])
                for i, c in zip(fr, cd_arr):
                    cd[i] = c
            # 生成子代
            children = []
            while len(children) < self.pop_size:
                def tour():
                    a, b = self.rng.integers(0, self.pop_size, 2)
                    if rank[a] < rank[b] or (rank[a] == rank[b] and cd[a] > cd[b]):
                        return a
                    return b
                p1 = X[tour()].copy(); p2 = X[tour()].copy()
                if self.rng.random() < self.pc:
                    child = _sbx(p1, p2, self.eta_c, self.rng)
                else:
                    child = p1.copy()
                child = _polynomial_mutation(child, lo, hi, self.pm, self.eta_m, self.rng)
                children.append(child)
            C = np.array(children)
            C = self._repair(C)
            FC = np.asarray(objective_fn(C))
            # 精英保留：父+子 合并后按非支配排序取前 pop_size
            X2 = np.vstack([X, C]); F2 = np.vstack([F, FC])
            fronts2 = fast_non_dominated_sort(F2)
            keep = []
            for fr in fronts2:
                if len(keep) + len(fr) <= self.pop_size:
                    keep.extend(fr)
                else:
                    cd2 = crowding_distance(F2[fr])
                    order = np.argsort(-cd2)
                    keep.extend([fr[i] for i in order[: self.pop_size - len(keep)]])
                    break
            X = X2[keep]; F = F2[keep]
        # 返回最终前沿（按目标值去重）
        pf = pareto_front_indices(F)
        Xf, Ff = X[pf], F[pf]
        _, uid = np.unique(np.round(Ff, 8), axis=0, return_index=True)
        return Xf[uid], Ff[uid]

if __name__ == "__main__":
    # 自检：ZDT1 风格 3 目标
    rng = np.random.default_rng(3)
    F = rng.random((200, 3))
    pf = pareto_front_indices(F)
    print("随机 200 点前沿大小:", len(pf))
    ns = NSGA2(pop_size=60, n_gen=50)
    def f(X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        return np.column_stack([x, y, z])
    Xf, Ff = ns.run(f)
    print("NSGA2 自检前沿大小:", len(Ff))
