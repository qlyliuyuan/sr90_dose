#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锶-90 (Sr-90 / Y-90) 皮肤敷贴器吸收剂量计算
================================================

计算目标
--------
给定敷贴器的核素(Sr-90)、活度(mCi)与标定日期，计算皮肤表面以下 3 mm 处
(水中/组织等效介质中) 的吸收剂量率 (Gy/s, Gy/min, Gy/h)。
若同时给定照射时间，则给出该深度处的累积吸收剂量 (Gy)。

本程序提供两种计算方法：
  1. method="pdd"（默认）: 经验 PDD 表 + 表面剂量率常数，快速估算。
  2. method="monte_carlo": 从 Sr-90/Y-90 β 能谱与 CSDA 射程出发做
     直线蒙特卡洛模拟，用于验证/教学。

输入输出说明
------------
题目要求输出“吸收剂量，单位为 Gy”。但仅由活度与标定日期只能得到剂量率；
要得到累积吸收剂量(Gy)，必须再提供照射时间。本程序通过参数
`exposure_time_seconds` 支持两种输出：
  - 未给照射时间：返回剂量率（多个常用单位）。
  - 给定照射时间：额外返回累积吸收剂量 `depth_dose_Gy` (Gy)。

物理依据
--------
1. 衰变链
   Sr-90 --β⁻ (Emax=0.546 MeV, T½=28.79 a)--> Y-90
   Y-90  --β⁻ (Emax=2.280 MeV, T½=64.10 h)--> Zr-90 (稳定)
   Y-90 的最大 β 能量远高于 Sr-90，且射程(~11 mm in water)覆盖 3 mm 深度，
   因此 3 mm 处的剂量几乎全部来自 Y-90。

2. 久期平衡 (secular equilibrium)
   Y-90 半衰期(64.1 h) 远小于 Sr-90(28.79 a)，故长期处于久期平衡：
   A(Y-90) = A(Sr-90)。整副敷贴器的剂量率随 Sr-90 的 28.79 a 半衰期衰减
   （而非 Y-90 的 64 h）。本程序据此做衰变校正。
   注：若敷贴器刚经化学分离(新去除 Y-90)，需约数周 Y-90 才能重新长回平衡；
   正常使用的敷贴器均满足久期平衡假设。

3. 表面剂量率
   D_dot_surf = A_now(GBq) × Γ_surf   [Gy/s]
   Γ_surf 强烈依赖敷贴器几何(活性面积、源窗厚度与材质、背衬)。
   临床上应使用厂家标定证书给出的 "标定日期表面吸收剂量率" —— 证书值优先！
   本程序提供一个基于能量平衡的物理估算默认值(见 SURFACE_DOSE_RATE_CONSTANT)，
   并支持直接传入证书表面剂量率以获得准确结果。

4. 深度剂量 (Percentage Depth Dose, PDD)
   D_dot(depth) = D_dot_surf × PDD(depth)
   PDD 采用 Sr-90/Y-90 在水中常见的典型深度剂量曲线(以表面为 100%)，
   并对 3 mm 做线性插值。不同敷贴器的实测 PDD 略有差异，应以实测为准。

参考文献方向
------------
- 核数据: DDEP/LNHB Nucléide, IAEA NuDat (Sr-90 T½=28.79 a; Y-90 T½=64.1 h,
  Emax=2.28 MeV, mean=0.933 MeV).
- β 射程: Y-90 在水中最大射程 ≈ 11 mm (Emax≈2.28 MeV).
- 深度剂量: Sr-90/Y-90 敷贴器水中 PDD 典型值 (IAEA/ICRU β 剂量学、
  Cross 等 β 点核及相关厂家资料中常见曲线)。

免责声明: 本程序用于教学/估算。临床使用必须以经计量溯源的标定证书实测值
为准，并由具备资质的医学物理人员核验。
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional, Union

# 蒙特卡洛后端为可选依赖；缺少 numpy 时 method="pdd" 仍可正常使用。
try:
    import sr90_monte_carlo as _mc
    _HAS_MC = _mc._HAS_NUMPY  # numpy 也必须可用，MC 才真正可用
except Exception:  # noqa: BLE001
    _mc = None  # type: ignore
    _HAS_MC = False

# ============================================================
# 1. 核数据 (Nuclear data, well-established)
# ============================================================
SR90_HALF_LIFE_YEARS = 28.79          # Sr-90 半衰期 (年)   DDEP/LNHB
Y90_HALF_LIFE_HOURS = 64.10           # Y-90 半衰期 (小时)  DDEP/LNHB
SR90_BETA_EMAX_MEV = 0.546            # Sr-90 β 最大能量 (MeV)
Y90_BETA_EMAX_MEV = 2.280             # Y-90 β 最大能量 (MeV)
SR90_BETA_EMEAN_MEV = 0.196           # Sr-90 β 平均能量 (MeV)
Y90_BETA_EMEAN_MEV = 0.933            # Y-90 β 平均能量 (MeV)
Y90_BETA_RANGE_WATER_MM = 11.0        # Y-90 β 在水中最大射程 (mm)

# ============================================================
# 2. 单位换算
# ============================================================
MCI_TO_BQ = 3.7e7                     # 1 mCi = 3.7×10⁷ Bq
BQ_TO_GBY = 1e-9                      # Bq -> GBq

# ============================================================
# 3. 表面吸收剂量率常数 (Surface dose-rate constant) —— 证书优先!
# ============================================================
# Γ_surf: 水中敷贴器表面吸收剂量率 / 活度  [Gy·s⁻¹·GBq⁻¹]
#
# !!! 强烈依赖敷贴器几何(活性面积/窗厚/背衬)，不同型号差异可达数倍 !!!
#     临床应以厂家标定证书值替换:  Γ_surf = (标定日表面剂量率 Gy/s) / (标定日活度 GBq)
#
# 默认值由能量平衡估算得到(典型皮肤敷贴器, 活性面积≈2 cm²):
#   每 Sr-90 衰变(平衡时)释放 β 平均能量 ≈ 0.196 + 0.933 = 1.129 MeV = 1.81e-13 J
#   前向进入组织份额≈0.5, 有效吸收深度≈0.2 cm, 面积≈2 cm², ρ=1 g/cm³
#   => 表面剂量率 ≈ 0.3 Gy/s per GBq, 估算不确定度约 ±(30~50)%。
SURFACE_DOSE_RATE_CONSTANT = 0.30     # Gy·s⁻¹·GBq⁻¹ (估算默认值)

# ============================================================
# 4. 深度剂量 PDD (水中, 表面=1.00) —— Sr-90/Y-90 典型曲线
# ============================================================
PDD_DEPTH_MM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
PDD_VALUE    = [1.00, 0.86, 0.74, 0.60, 0.49, 0.39, 0.31,
                0.20, 0.12, 0.075, 0.045, 0.025, 0.012, 0.004, 0.000]

TARGET_DEPTH_MM = 3.0                 # 本题目标深度


# ============================================================
# 工具函数
# ============================================================
def _to_date(d: Union[str, date, datetime]) -> date:
    """把字符串/ datetime 转为 date。"""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d.strip(), "%Y-%m-%d").date()
    raise TypeError(f"不支持的日期类型: {type(d)}")


def decay_factor(calibration_date: Union[str, date],
                 use_date: Optional[Union[str, date]] = None) -> tuple[float, float]:
    """按 Sr-90 半衰期计算衰变因子。

    久期平衡下整副敷贴器剂量率随 Sr-90 (28.79 a) 衰减。
    返回 (decay_factor, dt_years)。
    """
    cal = _to_date(calibration_date)
    use = _to_date(use_date) if use_date is not None else date.today()
    if use < cal:
        raise ValueError("使用日期早于标定日期，请检查输入。")
    dt_years = (use - cal).days / 365.25
    factor = 0.5 ** (dt_years / SR90_HALF_LIFE_YEARS)
    return factor, dt_years


def _interp_linear(x: float, xs: list[float], ys: list[float]) -> float:
    """线性插值，超出范围则按端点截断。"""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def pdd_at_depth(depth_mm: float) -> float:
    """返回指定深度(mm, 水中)的深度剂量(以表面为 1.0)。"""
    return _interp_linear(depth_mm, PDD_DEPTH_MM, PDD_VALUE)


# ============================================================
# 核心计算
# ============================================================
def compute_dose(
    activity_mci: float,
    calibration_date: Union[str, date],
    depth_mm: float = TARGET_DEPTH_MM,
    use_date: Optional[Union[str, date]] = None,
    surface_dose_rate_at_cal: Optional[float] = None,
    gamma_surf: float = SURFACE_DOSE_RATE_CONSTANT,
    exposure_time_seconds: Optional[float] = None,
    method: str = "pdd",
    mc_source_radius_mm: float = 10.0,
    mc_n_histories: int = 200_000,
    mc_dz_mm: float = 0.1,
    mc_tally_radius_mm: float = 2.0,
    mc_seed: Optional[int] = None,
) -> dict:
    """计算敷贴器在指定深度处的吸收剂量率 / 累积吸收剂量。

    重要输入输出说明
    ----------------
    由于输入里未包含照射时间，直接返回的是“吸收剂量率” (Gy/s, Gy/min)。
    若需要题目要求的“吸收剂量 (Gy)”，必须额外提供 `exposure_time_seconds`；
    此时程序会返回 `depth_dose_Gy`。

    方法选择
    --------
    method="pdd"（默认）: 使用经验 PDD 表 + 表面剂量率常数 Γ_surf。
    method="monte_carlo": 使用 Sr-90/Y-90 直线 CSDA 蒙特卡洛从能谱和射程出发
    直接计算剂量率。此时 `gamma_surf` 与 `surface_dose_rate_at_cal` 不再被使用，
    结果由 MC 的源几何与历史数决定。

    参数
    ----
    activity_mci : float
        标定日期时的活度 (mCi)。
    calibration_date : str | date
        标定日期, 如 "2020-01-01"。
    depth_mm : float
        目标深度 (mm, 皮肤表面以下), 默认 3.0。
    use_date : str | date | None
        使用(计算)日期, 默认今天。
    surface_dose_rate_at_cal : float | None
        标定日期的表面吸收剂量率 (Gy/s) —— 证书实测值。给出则优先使用，
        可消除 gamma_surf 的几何不确定性。
    gamma_surf : float
        表面剂量率常数 (Gy·s⁻¹·GBq⁻¹), 仅当未给出证书值且 method="pdd" 时使用。
    exposure_time_seconds : float | None
        照射时间 (秒)。给出则计算累积吸收剂量 (Gy); 否则仅返回剂量率。
    method : str
        计算方法: "pdd" (默认, 经验表) 或 "monte_carlo" (蒙特卡洛)。
    mc_source_radius_mm : float
        MC 平面圆盘源半径 (mm), 默认 10 mm。
    mc_n_histories : int
        MC 模拟历史数, 默认 200000。
    mc_dz_mm : float
        MC 深度分箱厚度 (mm), 默认 0.1 mm。
    mc_tally_radius_mm : float
        MC 中心轴 tally 圆柱半径 (mm), 默认 2 mm。
    mc_seed : int | None
        MC 随机数种子, 用于结果可复现。

    返回
    ----
    dict, 含活度、表面剂量率、深度剂量因子、目标深度剂量率(多种单位)
    及(若给定时间)累积吸收剂量。method="monte_carlo" 时还会返回
    `mc_rel_uncertainty` 与 `mc_pdd`。
    """
    # --- 输入校验 ---
    if activity_mci <= 0:
        raise ValueError("活度 activity_mci 必须为正数 (mCi)。")
    if depth_mm < 0:
        raise ValueError("深度 depth_mm 不能为负。")
    if exposure_time_seconds is not None and exposure_time_seconds < 0:
        raise ValueError("照射时间 exposure_time_seconds 不能为负。")
    if surface_dose_rate_at_cal is not None and surface_dose_rate_at_cal < 0:
        raise ValueError("证书表面剂量率 surface_dose_rate_at_cal 不能为负。")
    if method not in ("pdd", "monte_carlo"):
        raise ValueError("method 必须是 'pdd' 或 'monte_carlo'。")
    if method == "monte_carlo" and not _HAS_MC:
        raise ImportError(
            "method='monte_carlo' 需要 numpy 与 sr90_monte_carlo 模块。"
            "请安装 numpy 或改回 method='pdd'。"
        )

    # --- 衰变校正 ---
    use_date_eff = _to_date(use_date if use_date is not None else date.today())
    factor, dt_years = decay_factor(calibration_date, use_date_eff)
    activity_now_mci = activity_mci * factor
    activity_now_bq = activity_now_mci * MCI_TO_BQ
    activity_now_gby = activity_now_bq * BQ_TO_GBY

    # --- 表面/深度剂量率 ---
    if method == "monte_carlo":
        mc_res = _mc.mc_dose_rate_at_depth(
            activity_bq=activity_now_bq,
            depth_mm=depth_mm,
            source_radius_mm=mc_source_radius_mm,
            dz_mm=mc_dz_mm,
            tally_radius_mm=mc_tally_radius_mm,
            n_histories=mc_n_histories,
            seed=mc_seed,
        )
        surface_rate_now = mc_res["surface_rate_Gy_per_s"]
        depth_rate_now = mc_res["dose_rate_Gy_per_s"]
        pdd = mc_res["pdd"]
        mc_rel_uncertainty = mc_res["rel_uncertainty"]
        gamma_used = (
            surface_rate_now / activity_now_gby if activity_now_gby > 0 else None
        )
        source = (
            f"Monte Carlo 直线 CSDA (源半径={mc_source_radius_mm}mm, "
            f"N={mc_n_histories}, tally半径={mc_tally_radius_mm}mm)"
        )
    elif surface_dose_rate_at_cal is not None:
        # 证书实测值: 整副剂量率∝活度, 同样按 Sr-90 半衰期衰减
        surface_rate_now = surface_dose_rate_at_cal * factor
        gamma_used = surface_dose_rate_at_cal / (activity_mci * MCI_TO_BQ * BQ_TO_GBY)
        source = "证书实测表面剂量率 (推荐)"
        pdd = pdd_at_depth(depth_mm)
        mc_rel_uncertainty = None
        depth_rate_now = surface_rate_now * pdd
    else:
        surface_rate_now = activity_now_gby * gamma_surf
        gamma_used = gamma_surf
        source = f"常数估算 Γ_surf={gamma_surf} Gy·s⁻¹·GBq⁻¹ (几何相关, 证书优先)"
        pdd = pdd_at_depth(depth_mm)
        mc_rel_uncertainty = None
        depth_rate_now = surface_rate_now * pdd

    result = {
        "nuclide": "Sr-90 / Y-90",
        "method": method,
        "calibration_date": _to_date(calibration_date).isoformat(),
        "use_date": use_date_eff.isoformat(),
        "elapsed_years": dt_years,
        "decay_factor": factor,
        "activity_cal_mci": activity_mci,
        "activity_now_mci": activity_now_mci,
        "activity_now_gby": activity_now_gby,
        "surface_rate_source": source,
        "gamma_surf_used": gamma_used,
        "surface_rate_Gy_per_s": surface_rate_now,
        "surface_rate_Gy_per_min": surface_rate_now * 60.0,
        "depth_mm": depth_mm,
        "pdd": pdd,
        "depth_rate_Gy_per_s": depth_rate_now,
        "depth_rate_Gy_per_min": depth_rate_now * 60.0,
        "depth_rate_Gy_per_h": depth_rate_now * 3600.0,
    }

    if method == "monte_carlo":
        result["mc_rel_uncertainty"] = mc_rel_uncertainty

    if exposure_time_seconds is not None:
        result["exposure_time_s"] = exposure_time_seconds
        result["depth_dose_Gy"] = depth_rate_now * exposure_time_seconds
        result["surface_dose_Gy"] = surface_rate_now * exposure_time_seconds

    return result


# ============================================================
# 输出格式化
# ============================================================
def format_report(r: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  锶-90 (Sr-90/Y-90) 敷贴器吸收剂量计算结果")
    lines.append("=" * 60)
    lines.append(f"核素:              {r['nuclide']}")
    lines.append(f"计算方法:          {r['method']}")
    lines.append(f"标定日期:          {r['calibration_date']}")
    lines.append(f"计算日期:          {r['use_date']}")
    lines.append(f"经过时间:          {r['elapsed_years']:.3f} 年")
    lines.append(f"衰变因子:          {r['decay_factor']:.4f}  "
                 f"(按 Sr-90 T½={SR90_HALF_LIFE_YEARS} a)")
    lines.append("-" * 60)
    lines.append(f"标定活度:          {r['activity_cal_mci']:.3f} mCi")
    lines.append(f"当前活度:          {r['activity_now_mci']:.3f} mCi "
                 f"= {r['activity_now_gby']:.4f} GBq")
    lines.append("-" * 60)
    lines.append(f"表面剂量率来源:    {r['surface_rate_source']}")
    if r["gamma_surf_used"] is not None:
        lines.append(f"等效 Γ_surf:       {r['gamma_surf_used']:.4e} Gy·s⁻¹·GBq⁻¹")
    lines.append(f"表面吸收剂量率:    {r['surface_rate_Gy_per_s']:.4e} Gy/s "
                 f"= {r['surface_rate_Gy_per_min']:.3f} Gy/min")
    lines.append("-" * 60)
    lines.append(f"目标深度:          {r['depth_mm']:.1f} mm  (PDD={r['pdd']:.3f})")
    lines.append(f"  ★ {r['depth_mm']:.1f} mm 处剂量率: "
                 f"{r['depth_rate_Gy_per_s']:.4e} Gy/s")
    lines.append(f"                       = {r['depth_rate_Gy_per_min']:.4f} Gy/min")
    lines.append(f"                       = {r['depth_rate_Gy_per_h']:.4f} Gy/h")
    if "mc_rel_uncertainty" in r and r["mc_rel_uncertainty"] is not None:
        lines.append(f"  MC 统计相对不确定度: {r['mc_rel_uncertainty']:.2%}")
    if "depth_dose_Gy" in r:
        lines.append("-" * 60)
        lines.append(f"照射时间:          {r['exposure_time_s']:.1f} s")
        lines.append(f"  ★ {r['depth_mm']:.1f} mm 处累积吸收剂量: "
                     f"{r['depth_dose_Gy']:.4f} Gy")
        lines.append(f"     (表面累积吸收剂量: {r['surface_dose_Gy']:.4f} Gy)")
    else:
        lines.append("-" * 60)
        lines.append("注: 未给定照射时间 → 以上为剂量率(单位时间剂量)。")
        lines.append("    累积吸收剂量(Gy) = 深度剂量率(Gy/s) × 照射时间(s)。")
        lines.append("    调用 compute_dose(..., exposure_time_seconds=t) 可得累积剂量。")
    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================
# 示例 / CLI
# ============================================================
def _demo() -> None:
    print("=" * 60)
    print("  锶-90 (Sr-90/Y-90) 皮肤敷贴器剂量计算")
    print("=" * 60)

    # --- 交互输入 ---
    while True:
        try:
            activity = float(input("请输入放射源活度（mCi）：").strip())
            if activity <= 0:
                print("  活度必须为正数，请重新输入。")
                continue
            break
        except ValueError:
            print("  输入格式有误，请输入数字（如 50 或 50.5）。")

    while True:
        cal_date = input("请输入标定日期（格式 YYYY-MM-DD，如 2022-06-01）：").strip()
        try:
            _to_date(cal_date)
            break
        except (ValueError, TypeError):
            print("  日期格式有误，请按 YYYY-MM-DD 格式输入。")

    exp_input = input("请输入照射时间（秒，留空则只显示剂量率）：").strip()
    exposure = None
    if exp_input:
        try:
            exposure = float(exp_input)
            if exposure < 0:
                print("  照射时间不能为负，已忽略。")
                exposure = None
        except ValueError:
            print("  照射时间格式有误，已忽略。")

    print()

    # --- PDD 法 ---
    r = compute_dose(
        activity_mci=activity,
        calibration_date=cal_date,
        depth_mm=TARGET_DEPTH_MM,
        exposure_time_seconds=exposure,
        method="pdd",
    )
    print("【PDD 经验表法】")
    print(format_report(r))

    # --- Monte Carlo 法（如有 numpy）---
    if _HAS_MC:
        print()
        print("【Monte Carlo 直线 CSDA 法】")
        r_mc = compute_dose(
            activity_mci=activity,
            calibration_date=cal_date,
            depth_mm=TARGET_DEPTH_MM,
            exposure_time_seconds=exposure,
            method="monte_carlo",
            mc_n_histories=100_000,
            mc_seed=42,
        )
        print(format_report(r_mc))


def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="锶-90 敷贴器 3mm 处吸收剂量计算")
    p.add_argument("-a", "--activity", type=float, required=True,
                   help="标定日期活度 (mCi)")
    p.add_argument("-d", "--calibration-date", required=True,
                   help="标定日期 YYYY-MM-DD")
    p.add_argument("-u", "--use-date", default=None,
                   help="计算日期 YYYY-MM-DD (默认今天)")
    p.add_argument("--depth", type=float, default=TARGET_DEPTH_MM,
                   help="目标深度 mm (默认 3.0)")
    p.add_argument("--method", type=str, default="pdd",
                   choices=["pdd", "monte_carlo"],
                   help="计算方法: pdd (默认) 或 monte_carlo")
    p.add_argument("--gamma", type=float, default=SURFACE_DOSE_RATE_CONSTANT,
                   help="表面剂量率常数 Gy/s/GBq (默认估算值，仅 pdd 法)")
    p.add_argument("--surface-rate", type=float, default=None,
                   help="证书实测表面剂量率 Gy/s (标定日), 优先使用，仅 pdd 法")
    p.add_argument("-t", "--time", type=float, default=None,
                   help="照射时间 (秒), 给出则计算累积吸收剂量")
    # Monte Carlo 参数
    p.add_argument("--source-radius", type=float, default=10.0,
                   help="MC 源半径 mm (默认 10)")
    p.add_argument("--n-histories", type=int, default=200_000,
                   help="MC 历史数 (默认 200000)")
    p.add_argument("--dz", type=float, default=0.1,
                   help="MC 深度分箱 mm (默认 0.1)")
    p.add_argument("--tally-radius", type=float, default=2.0,
                   help="MC 中心轴 tally 半径 mm (默认 2)")
    p.add_argument("--seed", type=int, default=None,
                   help="MC 随机数种子")
    args = p.parse_args()
    r = compute_dose(
        activity_mci=args.activity,
        calibration_date=args.calibration_date,
        depth_mm=args.depth,
        use_date=args.use_date,
        method=args.method,
        surface_dose_rate_at_cal=args.surface_rate,
        gamma_surf=args.gamma,
        exposure_time_seconds=args.time,
        mc_source_radius_mm=args.source_radius,
        mc_n_histories=args.n_histories,
        mc_dz_mm=args.dz,
        mc_tally_radius_mm=args.tally_radius,
        mc_seed=args.seed,
    )
    print(format_report(r))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:
        _cli()
    else:
        _demo()
