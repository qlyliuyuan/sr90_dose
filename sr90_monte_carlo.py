#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sr-90/Y-90 敷贴器剂量计算 —— 蒙特卡洛后端（直线 CSDA 演示级）
=================================================================

本模块用蒙特卡洛方法模拟 Sr-90/Y-90 平面圆盘源在水中产生的 β 剂量。
采用用户选择的“直线 CSDA + numpy 加速”方案：
  - 电子从抽样点开始沿直线飞行，飞行距离等于其 CSDA 射程；
  - 能量按射程均匀沉积（CSDA 近似）；
  - 抽样 Sr-90 与 Y-90 的允许型 β 能谱，并按 1:1 混合（久期平衡）。

剂量在“以中心轴为轴心、半径为 tally_radius_mm 的细长圆柱”中统计，
因此输出近似为中心轴附近小区域内的剂量率，与经验 PDD 表的“中心轴相对深度剂量”对应。

物理假设与限制
----------------
1. 直线 CSDA：忽略多重散射与射程歧离，射程末端无散射拖尾；
   对 3 mm 处剂量率属于演示级估算，结果通常偏“硬”（射程末端锐利）。
2. 忽略源窗、背衬、空气间隙；源视为位于水/组织表面。
3. 电子能量低于截止能量时剩余能量就地沉积。
4. 横向采用半无限水模体（忽略侧向边界逃逸）。
"""

from __future__ import annotations

from typing import Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise ModuleNotFoundError(
            "蒙特卡洛模块需要 numpy。请先安装：\n"
            "  pip install numpy\n"
            "或使用默认的 PDD 经验表方法（无需 numpy）：\n"
            "  compute_dose(..., method='pdd')"
        )


# ---------------------------------------------------------------------------
# 物理常量
# ---------------------------------------------------------------------------
ELECTRON_REST_MASS_MEV = 0.51099895   # MeV/c²
MEV_TO_JOULE = 1.602176634e-13        # J/MeV
WATER_DENSITY_G_PER_CM3 = 1.0         # g/cm³

SR90_Q_MEV = 0.546
Y90_Q_MEV = 2.280

# 模块级缓存：numpy 可用时预计算，否则置 None，首次调用时再初始化
_SR90_SPEC_MAX: Optional[float] = None
_Y90_SPEC_MAX: Optional[float] = None


# ---------------------------------------------------------------------------
# 内部工具（仅在 numpy 可用时调用）
# ---------------------------------------------------------------------------
def _spectrum_pdf(energy_kin_mev: "np.ndarray", q_mev: float) -> "np.ndarray":
    """允许型 β 衰变谱的概率密度（未归一化）。"""
    m = ELECTRON_REST_MASS_MEV
    e_total = energy_kin_mev + m
    momentum = np.sqrt(energy_kin_mev * (energy_kin_mev + 2.0 * m))
    return momentum * e_total * np.maximum(q_mev - energy_kin_mev, 0.0) ** 2


def _find_spectrum_max(q_mev: float) -> float:
    """用网格搜索估计能谱最大值，用于拒绝抽样上界。"""
    e_grid = np.linspace(1e-6, q_mev - 1e-6, 5000)
    return float(np.max(_spectrum_pdf(e_grid, q_mev))) * 1.05


def _ensure_spec_max() -> None:
    """首次调用时懒初始化能谱最大值缓存（需要 numpy）。"""
    global _SR90_SPEC_MAX, _Y90_SPEC_MAX
    if _SR90_SPEC_MAX is None:
        _SR90_SPEC_MAX = _find_spectrum_max(SR90_Q_MEV)
    if _Y90_SPEC_MAX is None:
        _Y90_SPEC_MAX = _find_spectrum_max(Y90_Q_MEV)


# ---------------------------------------------------------------------------
# 能谱抽样
# ---------------------------------------------------------------------------
def sample_beta_energy(q_mev: float, n: int, rng: "np.random.Generator") -> "np.ndarray":
    """用拒绝抽样从允许型 β 能谱中抽取 n 个动能 (MeV)。"""
    _require_numpy()
    _ensure_spec_max()
    f_max = _SR90_SPEC_MAX if abs(q_mev - SR90_Q_MEV) < 1e-6 else _Y90_SPEC_MAX
    out = np.empty(n, dtype=np.float64)
    generated = 0
    while generated < n:
        batch = min(n - generated, 200_000)
        e = rng.uniform(0.0, q_mev, size=batch)
        f = _spectrum_pdf(e, q_mev)
        u = rng.uniform(0.0, f_max, size=batch)
        accepted = e[u < f]
        m = min(len(accepted), n - generated)
        out[generated:generated + m] = accepted[:m]
        generated += m
    return out


# ---------------------------------------------------------------------------
# CSDA 射程-能量关系（电子在水中）
# ---------------------------------------------------------------------------
def csda_range_mm(energy_mev: "np.ndarray") -> "np.ndarray":
    """
    电子在水中的 CSDA 射程 (mm)。
    使用经验公式: R [g/cm²] = 0.412 * E^(1.265 - 0.0954 * ln E),
    其中 E 单位为 MeV，0.01 < E < 3 MeV。水的 ρ=1 g/cm³，故 cm 数与 g/cm² 数值相同。
    返回 mm。
    """
    _require_numpy()
    e = np.asarray(energy_mev, dtype=np.float64)
    e = np.where(e < 1e-6, 1e-6, e)
    exponent = 1.265 - 0.0954 * np.log(e)
    r_cm = 0.412 * (e ** exponent)
    return r_cm * 10.0  # cm -> mm


# ---------------------------------------------------------------------------
# 蒙特卡洛深度剂量计算
# ---------------------------------------------------------------------------
def mc_depth_dose(
    activity_bq: float,
    source_radius_mm: float = 10.0,
    z_max_mm: float = 12.0,
    dz_mm: float = 0.1,
    tally_radius_mm: float = 2.0,
    n_histories: int = 200_000,
    seed: Optional[int] = None,
    energy_cutoff_mev: float = 0.01,
) -> dict:
    """用直线 CSDA 蒙特卡洛计算 Sr-90/Y-90 平面圆盘源在水中的深度剂量率。

    参数
    ----
    activity_bq : float
        当前 Sr-90 活度 (Bq)。久期平衡下 Y-90 活度与之相等。
    source_radius_mm : float
        平面圆盘源半径 (mm)。
    z_max_mm : float
        水模体最大深度 (mm)。
    dz_mm : float
        深度分箱厚度 (mm)。
    tally_radius_mm : float
        中心轴圆柱 tally 的半径 (mm)。剂量在该圆柱内统计。
    n_histories : int
        模拟历史数（每个历史代表一次 Sr-90 衰变，按 1:1 抽 Sr-90/Y-90）。
    seed : int | None
        随机数种子，用于结果可复现。
    energy_cutoff_mev : float
        能量截断 (MeV)，低于此值的电子剩余能量就地沉积。

    返回
    ----
    dict，包含:
      - depth_mm: 各 bin 中心深度 (mm)
      - dose_rate_Gy_per_s: 各 bin 剂量率 (Gy/s)
      - rel_uncertainty: 各 bin 相对统计不确定度
      - surface_rate_Gy_per_s: MC 计算的表面剂量率 (Gy/s)
      - pdd: 相对深度剂量 (表面归一化为 1.0)
      - bin_info: {dz_mm, tally_radius_mm, n_histories, ...}
    """
    _require_numpy()
    if activity_bq <= 0:
        raise ValueError("activity_bq 必须为正数。")
    if source_radius_mm <= 0 or tally_radius_mm <= 0:
        raise ValueError("源半径和 tally 半径必须为正数。")
    if dz_mm <= 0 or z_max_mm <= dz_mm:
        raise ValueError("dz_mm 必须为正且小于 z_max_mm。")
    if n_histories <= 0:
        raise ValueError("n_histories 必须为正整数。")

    rng = np.random.default_rng(seed)

    # 深度 bin 边界与中心
    z_edges = np.arange(0.0, z_max_mm + dz_mm * 0.5, dz_mm)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    n_bins = len(z_centers)

    # tally 圆柱体积 (cm³) -> 质量 (g)
    tally_radius_cm = tally_radius_mm * 0.1
    dz_cm = dz_mm * 0.1
    tally_volume_cm3 = np.pi * tally_radius_cm ** 2 * dz_cm
    tally_mass_g = tally_volume_cm3 * WATER_DENSITY_G_PER_CM3

    # 初始化统计量
    sum_e_dep = np.zeros(n_bins, dtype=np.float64)     # MeV
    sum_e_dep2 = np.zeros(n_bins, dtype=np.float64)    # MeV²
    n_scored = np.zeros(n_bins, dtype=np.int64)

    # 每个历史：按 1:1 抽样 Sr-90 / Y-90
    n_sr = n_histories // 2
    n_y = n_histories - n_sr

    def process_batch(batch_n: int, q_mev: float) -> None:
        nonlocal sum_e_dep, sum_e_dep2, n_scored
        if batch_n <= 0:
            return

        # 抽样能量
        e_mev = sample_beta_energy(q_mev, batch_n, rng)
        # 低于截止能量的直接沉积（射程极短，基本在表面 bin）
        # 为简化：仍计算其射程，让后续逻辑处理

        # 抽样源上位置（均匀圆盘）
        u_r = rng.random(batch_n)
        u_phi = 2.0 * np.pi * rng.random(batch_n)
        r_src = source_radius_mm * np.sqrt(u_r)
        x0 = r_src * np.cos(u_phi)
        y0 = r_src * np.sin(u_phi)

        # 抽样发射方向：上半球各向同性 -> cosθ 在 [0,1] 均匀
        cos_theta = rng.random(batch_n)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        phi_dir = 2.0 * np.pi * rng.random(batch_n)
        ux = sin_theta * np.cos(phi_dir)
        uy = sin_theta * np.sin(phi_dir)
        uz = cos_theta

        # CSDA 射程
        r_mev = np.where(e_mev > energy_cutoff_mev, e_mev, energy_cutoff_mev)
        r_mm = csda_range_mm(r_mev)
        # 总飞行距离 = 射程 (mm)
        # 终点深度 z1 = R * cosθ
        z1 = r_mm * uz

        # 对每条径迹，找出穿过哪些 bin 并在 tally 圆柱内沉积能量
        # 使用向量化处理所有 batch 中的历史
        for i_bin in range(n_bins):
            z_low = float(z_edges[i_bin])
            z_high = float(z_edges[i_bin + 1])
            # 该 bin 与径迹的 z 区间交集；z_in 是标量（所有粒子共用），z_out 是数组
            z_in = z_low                          # 起点永远是 z_low（所有轨迹从 z=0 出发）
            z_out = np.minimum(z_high, z1)        # 数组
            dz_seg = np.maximum(z_out - z_in, 0.0)

            # 只处理有穿过的径迹
            mask = dz_seg > 0.0
            if not np.any(mask):
                continue

            # 段中点深度（z_in 为标量，广播到 z_out[mask] 的长度）
            z_mid = 0.5 * (z_in + z_out[mask])
            # 该深度处径迹到中心轴的距离（参数 t = z_mid / z1）
            t = z_mid / np.maximum(z1[mask], 1e-12)
            x_mid = x0[mask] + (r_mm[mask] * ux[mask]) * t
            y_mid = y0[mask] + (r_mm[mask] * uy[mask]) * t
            r_mid = np.sqrt(x_mid ** 2 + y_mid ** 2)

            #  tally 条件
            in_tally = r_mid < tally_radius_mm
            if not np.any(in_tally):
                continue

            # 沉积能量 = E * (段长度 / 射程) = E * dz_seg / z1
            # 注意 z1 = R * uz，所以 dz_seg / z1 = 实际路径在 bin 内占比
            dep = e_mev[mask] * dz_seg[mask] / np.maximum(z1[mask], 1e-12)
            dep_tally = dep[in_tally]

            n_this = int(np.sum(in_tally))
            e_total = float(np.sum(dep_tally))
            e2_total = float(np.sum(dep_tally ** 2))

            sum_e_dep[i_bin] += e_total
            sum_e_dep2[i_bin] += e2_total
            n_scored[i_bin] += n_this

    # 处理 Sr-90 和 Y-90 两个批量
    process_batch(n_sr, SR90_Q_MEV)
    process_batch(n_y, Y90_Q_MEV)

    # 归一化到剂量率 (Gy/s)
    # 每历史平均沉积能量 (MeV) / tally 质量 (g) -> MeV/g
    # 1 MeV/g = 1.602e-13 J / 1e-3 kg = 1.602e-10 Gy
    # 剂量率 = 每历史剂量 * activity_bq (decays/s)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_dep_per_history = sum_e_dep / n_histories  # MeV/decay
        dose_per_history_Gy = mean_dep_per_history / tally_mass_g * MEV_TO_JOULE / 1e-3
        dose_rate_Gy_per_s = dose_per_history_Gy * activity_bq

        # 统计不确定度
        # var(E) = (sum_E2/N - mean_E^2) * N/(N-1)
        # rel_unc = sqrt(var(E)/N) / mean_E
        mean_e = sum_e_dep / n_histories
        mean_e2 = sum_e_dep2 / n_histories
        var_e = np.where(n_histories > 1,
                         (mean_e2 - mean_e ** 2) * n_histories / (n_histories - 1),
                         0.0)
        var_e = np.maximum(var_e, 0.0)
        sem_e = np.sqrt(var_e / n_histories)
        rel_uncertainty = np.where(mean_e > 0.0, sem_e / mean_e, np.inf)

    # 相对深度剂量 PDD：以表面剂量率为 1.0
    surface_rate = float(dose_rate_Gy_per_s[0]) if n_bins > 0 else 0.0
    pdd = np.where(surface_rate > 0.0, dose_rate_Gy_per_s / surface_rate, 0.0)

    return {
        "depth_mm": z_centers,
        "dose_rate_Gy_per_s": dose_rate_Gy_per_s,
        "rel_uncertainty": rel_uncertainty,
        "surface_rate_Gy_per_s": surface_rate,
        "pdd": pdd,
        "n_scored": n_scored,
        "bin_info": {
            "dz_mm": dz_mm,
            "tally_radius_mm": tally_radius_mm,
            "n_histories": n_histories,
            "source_radius_mm": source_radius_mm,
            "z_max_mm": z_max_mm,
        },
    }


def mc_dose_rate_at_depth(
    activity_bq: float,
    depth_mm: float,
    source_radius_mm: float = 10.0,
    z_max_mm: float = 12.0,
    dz_mm: float = 0.1,
    tally_radius_mm: float = 2.0,
    n_histories: int = 200_000,
    seed: Optional[int] = None,
) -> dict:
    """
    计算指定深度处的 MC 剂量率（Gy/s）与相对 PDD。

    返回
    ----
    dict: {depth_mm, dose_rate_Gy_per_s, pdd, rel_uncertainty,
           surface_rate_Gy_per_s, bin_info}
    """
    _require_numpy()
    if depth_mm < 0 or depth_mm > z_max_mm:
        raise ValueError(f"depth_mm 必须在 [0, {z_max_mm}] 范围内。")

    result = mc_depth_dose(
        activity_bq=activity_bq,
        source_radius_mm=source_radius_mm,
        z_max_mm=z_max_mm,
        dz_mm=dz_mm,
        tally_radius_mm=tally_radius_mm,
        n_histories=n_histories,
        seed=seed,
    )

    # 在目标深度处插值
    depths = result["depth_mm"]
    rates = result["dose_rate_Gy_per_s"]
    pdds = result["pdd"]
    unc = result["rel_uncertainty"]

    # 找到相邻 bin 做线性插值
    idx = int(np.searchsorted(depths, depth_mm))
    if idx <= 0:
        d_rate = float(rates[0])
        pdd_val = float(pdds[0])
        unc_val = float(unc[0])
    elif idx >= len(depths):
        d_rate = float(rates[-1])
        pdd_val = float(pdds[-1])
        unc_val = float(unc[-1])
    else:
        z0, z1 = depths[idx - 1], depths[idx]
        w = (depth_mm - z0) / (z1 - z0)
        d_rate = float(rates[idx - 1] * (1.0 - w) + rates[idx] * w)
        pdd_val = float(pdds[idx - 1] * (1.0 - w) + pdds[idx] * w)
        unc_val = float(unc[idx - 1] * (1.0 - w) + unc[idx] * w)

    return {
        "depth_mm": depth_mm,
        "dose_rate_Gy_per_s": d_rate,
        "pdd": pdd_val,
        "rel_uncertainty": unc_val,
        "surface_rate_Gy_per_s": result["surface_rate_Gy_per_s"],
        "bin_info": result["bin_info"],
    }


# ---------------------------------------------------------------------------
# 简单自测 / 示例
# ---------------------------------------------------------------------------
def _demo() -> None:
    """小示例：用 MC 估算 1 GBq Sr-90/Y-90 在 3 mm 处的剂量率。"""
    activity_bq = 1.0e9
    depth_mm = 3.0
    n_hist = 200_000
    print(f"运行 {n_hist} 历史的 Sr-90/Y-90 直线 CSDA MC ...")
    r = mc_dose_rate_at_depth(
        activity_bq=activity_bq,
        depth_mm=depth_mm,
        n_histories=n_hist,
        seed=42,
    )
    print(f"表面剂量率: {r['surface_rate_Gy_per_s']:.4e} Gy/s")
    print(f"{depth_mm} mm 处剂量率: {r['dose_rate_Gy_per_s']:.4e} Gy/s")
    print(f"{depth_mm} mm 处相对 PDD: {r['pdd']:.4f}")
    print(f"统计相对不确定度: {r['rel_uncertainty']:.3%}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    _demo()
