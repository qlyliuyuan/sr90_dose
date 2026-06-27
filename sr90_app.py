#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锶-90 皮肤敷贴器剂量计算 —— Streamlit 网页版
"""

import datetime
import streamlit as st
from sr90_dose_calculator import compute_dose, TARGET_DEPTH_MM, _HAS_MC

st.set_page_config(
    page_title="锶-90 敷贴器剂量计算",
    page_icon="☢️",
    layout="centered",
)

st.title("☢️ 锶-90 皮肤敷贴器剂量计算")
st.caption("Sr-90 / Y-90 平面源 · 皮肤表面以下 3 mm 处吸收剂量")

# ── 输入区 ────────────────────────────────────────────────
st.header("输入参数")

col1, col2 = st.columns(2)
with col1:
    activity = st.number_input(
        "放射源活度（mCi）",
        min_value=0.1, max_value=10000.0,
        value=50.0, step=0.1, format="%.2f",
    )
with col2:
    cal_date = st.date_input(
        "标定日期",
        value=datetime.date(2020, 1, 1),
        min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today(),
    )

col3, col4 = st.columns(2)
with col3:
    depth = st.number_input(
        "目标深度（mm）",
        min_value=0.0, max_value=11.0,
        value=float(TARGET_DEPTH_MM), step=0.5, format="%.1f",
    )
with col4:
    exposure = st.number_input(
        "照射时间（秒，0 表示只显示剂量率）",
        min_value=0.0, max_value=3600.0,
        value=0.0, step=1.0, format="%.1f",
    )

method = st.radio(
    "计算方法",
    options=["pdd", "monte_carlo"] if _HAS_MC else ["pdd"],
    format_func=lambda x: "PDD 经验表法（快速）" if x == "pdd" else "Monte Carlo 直线 CSDA 法",
    horizontal=True,
)

mc_histories = 100_000
if method == "monte_carlo":
    mc_histories = st.select_slider(
        "MC 历史数（越大越精确，越慢）",
        options=[10_000, 50_000, 100_000, 200_000, 500_000],
        value=100_000,
    )

# ── 计算 ─────────────────────────────────────────────────
if st.button("开始计算", type="primary", use_container_width=True):
    with st.spinner("计算中..." if method == "pdd" else f"Monte Carlo 运行 {mc_histories:,} 历史，请稍候..."):
        try:
            r = compute_dose(
                activity_mci=activity,
                calibration_date=cal_date.isoformat(),
                depth_mm=depth,
                exposure_time_seconds=exposure if exposure > 0 else None,
                method=method,
                mc_n_histories=mc_histories,
                mc_seed=42,
            )
        except Exception as e:
            st.error(f"计算出错：{e}")
            st.stop()

    # ── 衰变信息 ─────────────────────────────────────────
    st.header("计算结果")
    st.subheader("衰变校正")
    c1, c2, c3 = st.columns(3)
    c1.metric("经过时间", f"{r['elapsed_years']:.2f} 年")
    c2.metric("衰变因子", f"{r['decay_factor']:.4f}")
    c3.metric("当前活度", f"{r['activity_now_mci']:.2f} mCi")

    # ── 剂量率 ───────────────────────────────────────────
    st.subheader(f"{depth:.1f} mm 处吸收剂量率")
    d1, d2, d3 = st.columns(3)
    d1.metric("Gy/s",  f"{r['depth_rate_Gy_per_s']:.4e}")
    d2.metric("Gy/min", f"{r['depth_rate_Gy_per_min']:.4f}")
    d3.metric("Gy/h",  f"{r['depth_rate_Gy_per_h']:.4f}")

    if method == "monte_carlo" and r.get("mc_rel_uncertainty") is not None:
        st.info(f"MC 统计相对不确定度：{r['mc_rel_uncertainty']:.2%}")

    # ── 累积剂量 ─────────────────────────────────────────
    if "depth_dose_Gy" in r:
        st.subheader("累积吸收剂量")
        e1, e2 = st.columns(2)
        e1.metric(
            f"{depth:.1f} mm 处（目标深度）",
            f"{r['depth_dose_Gy']:.4f} Gy",
        )
        e2.metric(
            "皮肤表面",
            f"{r['surface_dose_Gy']:.4f} Gy",
        )

    # ── 详细参数 ─────────────────────────────────────────
    with st.expander("详细参数"):
        st.table({
            "参数": [
                "核素", "计算方法", "标定日期", "计算日期",
                "标定活度 (mCi)", "当前活度 (GBq)",
                "表面剂量率来源", "PDD",
            ],
            "值": [
                r["nuclide"], r["method"],
                r["calibration_date"], r["use_date"],
                f"{r['activity_cal_mci']:.3f}",
                f"{r['activity_now_gby']:.4f}",
                r["surface_rate_source"],
                f"{r['pdd']:.4f}",
            ],
        })

    # ── 免责声明 ─────────────────────────────────────────
    st.warning(
        "⚠️ 本工具仅供教学与估算使用。"
        "临床剂量计算必须以经计量溯源的标定证书实测值为准，并由具备资质的医学物理人员核验。"
    )
