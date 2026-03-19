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
    """
    Returns dict like:
    {
      "1Y": {"cagr": float or None, "avg": float or None},
      ...
    }
    """
    if df.empty:
        return {f"{h}Y": {"cagr": None, "avg": None} for h in horizons_years}

    df = df.sort_values("Date")
    today = df["Date"].max()

    results = {}
    for h in horizons_years:
        # <= h years back, with small buffer
        start_cutoff = today - timedelta(days=h * 365 + h // 4)
        df_h = df[df["Date"] >= start_cutoff].copy()

        # need at least 2 points
        if df_h.shape[0] < 2:
            results[f"{h}Y"] = {"cagr": None, "avg": None}
            continue

        start_nav = df_h.iloc[0]["NAV"]
        end_nav = df_h.iloc[-1]["NAV"]

        # use actual years between first and last to be robust
        days_diff = (df_h.iloc[-1]["Date"] - df_h.iloc[0]["Date"]).days
        years_diff = max(days_diff / 365.25, 0.1)

        cagr = calc_cagr(start_nav, end_nav, years_diff)

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
    "XIRR shown here is approximated using NAV, not actual cash flows.[web:112][web:113]"
)

schemes_df = load_all_schemes_df()

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
        ann_vol = daily_std * np.sqrt(252) if not pd.isna(daily_std) else None
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

# Show raw table so you can see data is there
st.dataframe(perf_df, width=1000, height=300)

# ---------------- RISK FACTOR SECTION ----------------
st.markdown("---")
st.subheader("Risk Factor – Annualized Volatility for Selected Schemes")

risk_df = pd.DataFrame(risk_rows)
st.dataframe(risk_df, width="stretch")

risk_numeric = risk_df.copy()
risk_numeric["Risk_Value"] = pd.to_numeric(
    risk_numeric["Annualized Volatility (Risk)"].str.replace("%", "", regex=False),
    errors="coerce",
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
    "Higher values mean more variability and therefore higher risk.[web:118]"
)

# --------- Visual CAGR comparison: one bar chart per scheme (1,3,5,10Y) ---------
st.markdown("#### Visual CAGR comparison – 1Y, 3Y, 5Y, 10Y per fund")

cagr_numeric_df = perf_df.copy()
for h in HORIZONS_YEARS:
    col = f"{h}Y CAGR"
    cagr_numeric_df[col] = pd.to_numeric(
        cagr_numeric_df[col].str.replace("%", "", regex=False),
        errors="coerce",
    )

for _, row in cagr_numeric_df.iterrows():
    scheme_name = row["Scheme Name"]
    data_rows = []
    for h in HORIZONS_YEARS:
        col = f"{h}Y CAGR"
        val = row[col]
        # Only append if we have a numeric value
        if not pd.isna(val):
            data_rows.append({"Horizon": f"{h}Y", "CAGR": val})

    if not data_rows:
        st.info(f"No valid CAGR data for {scheme_name} across 1/3/5/10 years.")
        continue

    df_scheme = pd.DataFrame(data_rows)
    df_scheme["Label"] = df_scheme["CAGR"].round(2).astype(str) + "%"

    fig_scheme = px.bar(
        df_scheme,
        x="Horizon",
        y="CAGR",
        text="Label",
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
st.dataframe(fund_cagr_df, width=1000)

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
