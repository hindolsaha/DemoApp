import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Optional

st.set_page_config(
    page_title="MF Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---------------- CONFIG ----------------
MAX_FUNDS = 20
HORIZONS_YEARS = [1, 3, 5, 10]

FUND_TYPES = [
    "All",
    "Large Cap",
    "Mid Cap",
    "Small Cap",
    "Index Fund",
    "Flexi Cap",
    "Others",
]

DEFAULT_FUNDS = [
    "UTI Nifty Next 50 Index Fund - Direct Plan - Growth Option",
    "UTI Nifty 50 Index Fund - Growth Option- Direct",
    "DSP Nifty 50 Index Fund - Direct Plan - Growth",
    "DSP Nifty Next 50 Index Fund - Direct Plan - Growth",
    "Aditya Birla Sun Life Gold Fund - Growth - Direct Plan",
]

# ---------------- HELPERS ----------------
@st.cache_data(ttl=3600)
def load_all_schemes_df():
    url = "https://api.mfapi.in/mf"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json())

@st.cache_data(ttl=300)
def fetch_history_days(scheme_code: str, days: int = 365 * 10) -> pd.DataFrame:
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "data" not in data or not data["data"]:
            return pd.DataFrame()

        rows = data["data"]
        cutoff = datetime.today() - timedelta(days=days)
        parsed = []
        for row in rows:
            try:
                d = datetime.strptime(row["date"], "%d-%m-%Y")
                if d < cutoff:
                    break
                parsed.append({"Date": d, "NAV": float(row["nav"])})
            except Exception:
                continue

        if not parsed:
            return pd.DataFrame()

        df = pd.DataFrame(parsed).sort_values("Date")
        return df
    except Exception:
        return pd.DataFrame()

def calc_cagr(start_nav: float, end_nav: float, years: float) -> Optional[float]:
    if start_nav <= 0 or end_nav <= 0 or years <= 0:
        return None
    try:
        return (end_nav / start_nav) ** (1.0 / years) - 1.0
    except Exception:
        return None

def get_horizon_returns(df: pd.DataFrame, horizons_years=HORIZONS_YEARS):
    if df.empty:
        return {f"{h}Y": {"cagr": None, "avg": None} for h in horizons_years}

    today = df["Date"].max()
    results = {}
    for h in horizons_years:
        start_cutoff = today - timedelta(days=h * 365 + h // 4)
        df_h = df[df["Date"] >= start_cutoff].copy()
        if df_h.shape[0] < 2:
            results[f"{h}Y"] = {"cagr": None, "avg": None}
            continue

        start_nav = df_h.iloc[0]["NAV"]
        end_nav = df_h.iloc[-1]["NAV"]

        cagr = calc_cagr(start_nav, end_nav, float(h))

        df_h = df_h.sort_values("Date")
        df_h["daily_ret"] = df_h["NAV"].pct_change()
        avg_daily = df_h["daily_ret"].dropna().mean()
        if pd.isna(avg_daily):
            avg_annual = None
        else:
            avg_annual = (1.0 + avg_daily) ** 252 - 1.0

        results[f"{h}Y"] = {"cagr": cagr, "avg": avg_annual}

    return results

def fmt_pct(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"

def filter_by_fund_type(df: pd.DataFrame, fund_type: str) -> pd.DataFrame:
    if fund_type == "All":
        return df

    name = df["schemeName"].str.lower()

    if fund_type == "Large Cap":
        mask = name.str.contains("large cap", na=False)
    elif fund_type == "Mid Cap":
        mask = name.str.contains("mid cap", na=False)
    elif fund_type == "Small Cap":
        mask = name.str.contains("small cap", na=False)
    elif fund_type == "Index Fund":
        mask = name.str.contains("index", na=False)
    elif fund_type == "Flexi Cap":
        mask = name.str.contains("flexi", na=False)
    elif fund_type == "Others":
        mask_large = name.str.contains("large cap", na=False)
        mask_mid = name.str.contains("mid cap", na=False)
        mask_small = name.str.contains("small cap", na=False)
        mask_index = name.str.contains("index", na=False)
        mask_flexi = name.str.contains("flexi", na=False)
        mask = ~(mask_large | mask_mid | mask_small | mask_index | mask_flexi)
    else:
        return df

    return df[mask]

# ---------------- LOAD SCHEMES & SELECT FUNDS ----------------
st.title("Mutual Fund Multi-Fund Performance Dashboard")
st.caption(
    "Source: MFAPI.in – NAV-based approximations for 1, 3, 5, 10 year trends. "
    "XIRR shown here is approximated using NAV, not actual cash flows."
)

schemes_df = load_all_schemes_df()

# ---- Last data refresh status ----
st.markdown(
    f"**Data status:** Fetched from MFAPI.in on "
    f"{datetime.today().strftime('%d-%b-%Y %H:%M')} (local time)."
)

st.markdown("### Fund selection")

col_type, col_sel1, col_sel2 = st.columns([1.5, 2, 3])

with col_type:
    chosen_type = st.selectbox(
        "Fund type",
        options=FUND_TYPES,
        index=0,
        help="Filter schemes by basic category using name keywords.",
    )

filtered_df = filter_by_fund_type(schemes_df, chosen_type)

with col_sel1:
    search_text = st.text_input(
        "Search within selected fund type",
        value="",
        placeholder="Type part of name, e.g. 'quant', 'nifty', 'flexi cap'...",
    )

if search_text.strip():
    options_df = filtered_df[
        filtered_df["schemeName"].str.contains(search_text, case=False, na=False)
    ]
else:
    options_df = filtered_df

if options_df.empty:
    st.warning("No funds match your type + search filter.")
    st.stop()

fund_names = options_df["schemeName"].sort_values().tolist()

default_selection = [f for f in DEFAULT_FUNDS if f in fund_names]

with col_sel2:
    selected_names = st.multiselect(
        f"Choose up to {MAX_FUNDS} mutual funds",
        options=fund_names,
        default=default_selection if default_selection else fund_names[: min(len(fund_names), 3)],
        max_selections=MAX_FUNDS,
    )

if not selected_names:
    st.info("Select at least one fund to see dashboard.")
    st.stop()

selected_rows = options_df[options_df["schemeName"].isin(selected_names)].copy()
selected_rows["schemeCode"] = selected_rows["schemeCode"].astype(str)

st.markdown("---")

# ---------------- COMPUTE RETURNS & RISK FOR EACH FUND ----------------
st.subheader("1 / 3 / 5 / 10 Year Trend – CAGR & Avg Annual Return")

perf_rows = []
risk_rows = []
nav_histories = {}

for _, row in selected_rows.iterrows():
    code = row["schemeCode"]
    name = row["schemeName"]

    df_nav = fetch_history_days(code, days=365 * 10)
    nav_histories[code] = df_nav

    # Horizon-wise returns
    ret_map = get_horizon_returns(df_nav, horizons_years=HORIZONS_YEARS)

    data = {
        "Scheme Code": code,
        "Scheme Name": name,
    }
    for h in HORIZONS_YEARS:
        key = f"{h}Y"
        data[f"{key} CAGR"] = fmt_pct(ret_map[key]["cagr"])
        data[f"{key} Avg"] = fmt_pct(ret_map[key]["avg"])

    perf_rows.append(data)

    # Risk: annualized volatility from daily NAV returns
    if not df_nav.empty:
        df_r = df_nav.sort_values("Date").copy()
        df_r["daily_ret"] = df_r["NAV"].pct_change()
        daily_std = df_r["daily_ret"].dropna().std()
        if pd.isna(daily_std):
            ann_vol = None
        else:
            ann_vol = daily_std * np.sqrt(252)
    else:
        ann_vol = None

    risk_rows.append(
        {
            "Scheme Code": code,
            "Scheme Name": name,
            "Annualized Volatility (Risk)": fmt_pct(ann_vol),
        }
    )

perf_df = pd.DataFrame(perf_rows)

# ---------------- RISK FACTOR SECTION ----------------
st.markdown("---")
st.subheader("Risk Factor – Annualized Volatility for Selected Schemes")

risk_df = pd.DataFrame(risk_rows)
st.dataframe(risk_df, width="stretch")

risk_numeric = risk_df.copy()
risk_numeric["Risk_Value"] = (
    risk_numeric["Annualized Volatility (Risk)"]
    .str.replace("%", "", regex=False)
    .replace("N/A", np.nan)
    .astype(float)
)

fig_risk = px.bar(
    risk_numeric,
    x="Scheme Name",
    y="Risk_Value",
    title="Annualized Volatility (Risk) – Higher = More Volatile",
)
fig_risk.update_layout(
    yaxis_title="Annualized Volatility (%)",
    xaxis_title="Scheme",
    height=400,
)
st.plotly_chart(fig_risk, use_container_width=True, config={"responsive": True})

st.caption(
    "Risk is approximated here as annualized volatility (standard deviation of daily NAV returns). "
    "Higher values mean more variability and therefore higher risk."
)

# --------- Visual CAGR comparison: one bar chart per scheme (stacked vertically) ---------
st.markdown("#### Visual CAGR comparison (one chart below another)")

cagr_numeric_df = perf_df.copy()
for h in HORIZONS_YEARS:
    col = f"{h}Y CAGR"
    cagr_numeric_df[col] = (
        cagr_numeric_df[col]
        .str.replace("%", "", regex=False)
        .replace("N/A", np.nan)
        .astype(float)
    )

for _, row in cagr_numeric_df.iterrows():
    scheme_name = row["Scheme Name"]
    data_rows = []
    for h in HORIZONS_YEARS:
        col = f"{h}Y CAGR"
        val = row[col]
        data_rows.append({"Horizon": f"{h}Y", "CAGR": val})

    df_scheme = pd.DataFrame(data_rows)
    df_scheme["CAGR_label"] = df_scheme["CAGR"].round(2).astype(str) + "%"

    fig_scheme = px.bar(
        df_scheme,
        x="Horizon",
        y="CAGR",
        text="CAGR_label",
        title=f"{scheme_name} – CAGR for 1Y / 3Y / 5Y / 10Y",
    )

    fig_scheme.update_traces(
        textposition="outside",
        marker=dict(line=dict(width=1.5, color="black")),
    )

    fig_scheme.update_layout(
        yaxis_title="CAGR (%)",
        height=350,
        xaxis_title="Duration",
    )

    st.plotly_chart(fig_scheme, config={"responsive": True})

# ---------------- SIMPLE FUND RETURNS (NO BENCHMARK) ----------------
st.markdown("---")
st.subheader("Fund 1 / 3 / 5 / 10 Year CAGR (No Benchmark)")

cagr_cols = ["Scheme Code", "Scheme Name"] + [f"{h}Y CAGR" for h in HORIZONS_YEARS]
fund_cagr_df = perf_df[cagr_cols].copy()
st.dataframe(fund_cagr_df, width="stretch")

# ---------------- NAV TREND ----------------
st.markdown("---")
st.subheader("NAV Trend – Last 10 Years (if available)")

chart_list = []
for _, row in selected_rows.iterrows():
    code = row["schemeCode"]
    name = row["schemeName"]

    df_nav = nav_histories.get(code, pd.DataFrame())
    if df_nav.empty:
        continue
    df_plot = df_nav.copy()
    df_plot["Scheme Name"] = name
    chart_list.append(df_plot)

if chart_list:
    nav_all = pd.concat(chart_list, ignore_index=True)

    fig = px.line(
        nav_all,
        x="Date",
        y="NAV",
        color="Scheme Name",
        title="NAV History – Selected Funds (Smoothed)",
        line_shape="spline",
        render_mode="svg",
    )

    fig.update_traces(line=dict(width=2))

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        xaxis=dict(
            title="Date",
            tickangle=0,
            tickfont=dict(size=8),
            showgrid=False,
            nticks=6,
        ),
        yaxis=dict(
            title="NAV",
            tickfont=dict(size=8),
            showgrid=True,
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
else:
    st.info("No NAV history available for selected funds.")

# ---------------- YOY RETURNS ----------------
st.markdown("---")
st.subheader("Year-on-Year (YoY) NAV-based Returns (Approx XIRR)")

fund_for_yoy = st.selectbox(
    "Select a fund (from above selection) for detailed YoY returns",
    options=selected_rows["schemeName"].tolist(),
)

yoy_rows = []
for _, row in selected_rows[selected_rows["schemeName"] == fund_for_yoy].iterrows():
    code = row["schemeCode"]
    name = row["schemeName"]
    df_nav = nav_histories.get(code, pd.DataFrame())
    if df_nav.empty:
        continue

    df_nav = df_nav.copy()
    df_nav["Year"] = df_nav["Date"].dt.year

    grp = df_nav.groupby("Year")
    for yr, g in grp:
        if g.shape[0] < 2:
            continue
        g = g.sort_values("Date")
        start_nav = g.iloc[0]["NAV"]
        end_nav = g.iloc[-1]["NAV"]
        ret = end_nav / start_nav - 1.0
        yoy_rows.append(
            {
                "Scheme Name": name,
                "Year": int(yr),
                "YoY Return (approx XIRR)": fmt_pct(ret),
            }
        )

if yoy_rows:
    yoy_df = pd.DataFrame(yoy_rows).sort_values("Year", ascending=False)
    st.dataframe(yoy_df, width="stretch")
    st.caption(
        "YoY returns above are NAV-based approximations and not true XIRR "
        "(which requires actual cash-flow data from your investments)."
    )
else:
    st.info("Not enough data to compute YoY returns for the selected fund.")

# ---------------- ACTION SELECTION ----------------
st.markdown("---")
action = st.radio(
    "Choose what you want to do",
    options=[
        "Investment Projection – SIP / Lump Sum",
        "Goal-based Target Amount – Suggest Best 2 Funds (10Y CAGR)",
    ],
    index=0,
)

# ---------------- INVESTMENT PROJECTION ----------------
if action == "Investment Projection – SIP / Lump Sum":
    st.subheader("Investment Projection – SIP / Lump Sum (Tentative)")

    col_inv1, col_inv2, col_inv3 = st.columns(3)

    with col_inv1:
        proj_fund = st.selectbox(
            "Choose fund for projection",
            options=selected_rows["schemeName"].tolist(),
            key="proj_fund",
        )

    with col_inv2:
        invest_mode = st.radio(
            "Investment mode",
            options=["SIP (monthly)", "Lump sum"],
        )

    with col_inv3:
        proj_years = st.number_input(
            "Years to stay invested",
            min_value=1,
            max_value=40,
            value=10,
            step=1,
        )

    amount = st.number_input(
        "Investment amount (₹)",
        min_value=100.0,
        value=5000.0,
        step=500.0,
    )

    horizon_for_expected = st.selectbox(
        "Use which historical CAGR as expected return?",
        options=[f"{h}Y" for h in HORIZONS_YEARS],
        index=2,
    )

    proj_row = perf_df[
        perf_df["Scheme Name"].str.lower() == proj_fund.lower()
    ]

    if proj_row.empty:
        st.info("Please select a valid fund for projection.")
    else:
        cagr_str = proj_row.iloc[0][f"{horizon_for_expected} CAGR"]
        if cagr_str == "N/A":
            st.warning("No valid historical CAGR found for this horizon; cannot project.")
        else:
            expected_cagr = float(cagr_str.replace("%", "")) / 100.0
            r = expected_cagr
            n_years = proj_years

            if invest_mode == "Lump sum":
                final_value = amount * (1.0 + r) ** n_years
                gain_pct = (final_value / amount - 1.0) * 100.0

                st.markdown("#### Lump sum projection (tentative)")
                st.write(f"Expected CAGR used: **{cagr_str}** based on {horizon_for_expected} history.")
                st.write(f"Initial investment: **₹{amount:,.0f}**")
                st.write(f"Projected value after {n_years} years: **₹{final_value:,.0f}**")
                st.write(f"Total gain: **{gain_pct:.2f}%**")

                years_axis = list(range(0, n_years + 1))
                values = [amount * (1.0 + r) ** y for y in years_axis]
                df_proj = pd.DataFrame({"Year": years_axis, "Projected Value": values})
                fig_proj = px.line(
                    df_proj,
                    x="Year",
                    y="Projected Value",
                    title="Lump Sum Projection – Value Over Time",
                    markers=True,
                )
                fig_proj.update_layout(yaxis_tickprefix="₹", height=400)
                st.plotly_chart(fig_proj, config={"responsive": True})

            else:
                m = 12 * n_years
                i = (1.0 + r) ** (1.0 / 12.0) - 1.0
                if i <= 0:
                    final_value = amount * m
                else:
                    final_value = amount * (((1.0 + i) ** m - 1.0) / i) * (1.0 + i)
                total_invested = amount * m
                gain_pct = (final_value / total_invested - 1.0) * 100.0

                st.markdown("#### SIP projection (tentative)")
                st.write(f"Expected CAGR used: **{cagr_str}** based on {horizon_for_expected} history.")
                st.write(f"Monthly SIP: **₹{amount:,.0f}**, duration: **{n_years} years ({m} months)**")
                st.write(f"Total invested: **₹{total_invested:,.0f}**")
                st.write(f"Projected value after {n_years} years: **₹{final_value:,.0f}**")
                st.write(f"Total gain: **{gain_pct:.2f}%**")

                months = list(range(1, m + 1))
                values = []
                running_value = 0.0
                for _ in months:
                    if i <= 0:
                        running_value += amount
                    else:
                        running_value = running_value * (1.0 + i) + amount
                    values.append(running_value)

                df_sip = pd.DataFrame(
                    {
                        "Month": months,
                        "Projected Value": values,
                        "Total Invested": [amount * k for k in months],
                    }
                )

                fig_sip = px.line(
                    df_sip,
                    x="Month",
                    y=["Projected Value", "Total Invested"],
                    title="SIP Projection – Projected vs Invested Over Time",
                )
                fig_sip.update_layout(yaxis_tickprefix="₹", height=400)

                last_month = df_sip["Month"].iloc[-1]
                last_proj = df_sip["Projected Value"].iloc[-1]
                last_invested = df_sip["Total Invested"].iloc[-1]

                fig_sip.add_scatter(
                    x=[last_month],
                    y=[last_proj],
                    mode="markers+text",
                    text=[f"Projected: ₹{last_proj:,.0f}"],
                    textposition="top center",
                    marker=dict(color="green", size=10),
                    showlegend=False,
                )

                fig_sip.add_scatter(
                    x=[last_month],
                    y=[last_invested],
                    mode="markers+text",
                    text=[f"Invested: ₹{last_invested:,.0f}"],
                    textposition="bottom center",
                    marker=dict(color="orange", size=10),
                    showlegend=False,
                )

                st.plotly_chart(fig_sip, config={"responsive": True})

            st.caption(
                "These projections are purely illustrative, using past NAV-based CAGR as expected return. "
                "They are not guaranteed and not investment advice."
            )

# ---------------- GOAL-BASED FUND SUGGESTION (10Y CAGR) ----------------
if action == "Goal-based Target Amount – Suggest Best 2 Funds (10Y CAGR)":
    st.subheader("Goal-based Target Amount – Suggest Best 2 Funds (Using 10Y CAGR)")

    st.markdown(
        "Enter the **target amount** and **time frame**. "
        "The tool will use the **10 year historical CAGR (10Y CAGR)** of all selected funds "
        "to suggest the top 2 funds for this goal."
    )

    goal_mode = st.radio(
        "Goal mode",
        options=["Lump sum today", "Monthly SIP"],
        index=0,
        horizontal=True,
    )

    goal_col1, goal_col2 = st.columns(2)

    with goal_col1:
        target_amount = st.number_input(
            "Target amount needed (₹)",
            min_value=10000.0,
            value=10000000.0,
            step=50000.0,
        )

    with goal_col2:
        target_years = st.number_input(
            "Years available to reach goal",
            min_value=1,
            max_value=40,
            value=10,
            step=1,
        )

    closest_horizon = 10
    closest_col = "10Y CAGR"

    st.write(
        f"Using **10 year** historical CAGR column ({closest_col}) to evaluate funds for this goal."
    )

    perf_numeric = perf_df.copy()
    perf_numeric[closest_col] = (
        perf_numeric[closest_col]
        .str.replace("%", "", regex=False)
        .replace("N/A", np.nan)
        .astype(float)
        / 100.0
    )

    perf_10y = perf_numeric.dropna(subset=[closest_col])

    if perf_10y.empty:
        st.info("No valid 10Y CAGR data available for selected funds.")
    else:
        perf_sorted = perf_10y.sort_values(closest_col, ascending=False)
        top2 = perf_sorted.head(2).copy()

        results = []
        for _, r in top2.iterrows():
            name = r["Scheme Name"]
            code = r["Scheme Code"]
            ann_return = r[closest_col]
            n = float(target_years)

            if goal_mode == "Lump sum today":
                required_lump_sum = target_amount / ((1.0 + ann_return) ** n)
                results.append(
                    {
                        "Scheme Code": code,
                        "Scheme Name": name,
                        "Historical 10Y CAGR": fmt_pct(ann_return),
                        "Required Lump Sum Now (₹)": f"{required_lump_sum:,.0f}",
                    }
                )
            else:
                m = int(n * 12)
                i = (1.0 + ann_return) ** (1.0 / 12.0) - 1.0
                if i <= 0 or m <= 0:
                    required_sip = np.nan
                else:
                    factor = (((1.0 + i) ** m - 1.0) / i) * (1.0 + i)
                    required_sip = target_amount / factor if factor > 0 else np.nan

                results.append(
                    {
                        "Scheme Code": code,
                        "Scheme Name": name,
                        "Historical 10Y CAGR": fmt_pct(ann_return),
                        "Required Monthly SIP (₹)": f"{required_sip:,.0f}",
                    }
                )

        goal_df = pd.DataFrame(results)
        st.dataframe(goal_df, width="stretch")

        st.caption(
            "Above suggestions are based purely on 10-year NAV-based CAGR of selected funds "
            "and standard compound interest formulas. They are indicative only, not investment advice."
        )
