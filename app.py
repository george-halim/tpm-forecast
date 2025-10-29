# app.py
import io
import base64
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

st.set_page_config(page_title="Shipment Forecast Agent", layout="wide")
st.title("Shipment Forecast Agent — Monthly, Multi‑SKU, Auto-parse")

st.sidebar.header("Upload / Settings")
uploaded = st.sidebar.file_uploader("Upload Excel file (monthly data)", type=["xlsx", "xls"])
months_to_forecast = st.sidebar.number_input("Forecast months", min_value=1, max_value=24, value=12)
alert_threshold_percent = st.sidebar.number_input("Alert threshold vs IMS (%)", min_value=10, max_value=500, value=150)

def read_excel_bytes(bytes_io):
    try:
        xls = pd.ExcelFile(bytes_io)
        sheets = []
        for name in xls.sheet_names:
            s = pd.read_excel(xls, sheet_name=name, header=None)
            s["_sheet_name"] = name
            sheets.append(s)
        df_raw = pd.concat(sheets, ignore_index=True, sort=False)
        df_raw = df_raw.dropna(how="all").reset_index(drop=True)
        return df_raw
    except Exception:
        try:
            df = pd.read_excel(bytes_io, header=None)
            df = df.dropna(how="all").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

def try_transpose_vertical_block(df):
    if df.shape[1] == 0:
        return pd.DataFrame()
    first_cols = [c for c in df.columns if c != "_sheet_name"]
    if not first_cols:
        return pd.DataFrame()
    col0 = first_cols[0]
    items = df[col0].astype(str).map(lambda x: x.strip()).tolist()
    items = [i for i in items if i and i.lower() != "nan"]
    lower_items = [i.lower() for i in items]
    header_labels = ["month", "sku", "stock", "ims", "shipment", "shipments", "onhand", "inventory"]
    header_hits = sum(1 for it in lower_items[:40] if any(h in it for h in header_labels))
    if header_hits < 2 and df.shape[1] > 1:
        for rid in range(min(6, df.shape[0])):
            rowvals = df.iloc[rid].astype(str).str.strip().str.lower().tolist()
            if any(h in " ".join(rowvals) for h in header_labels):
                header = df.iloc[rid].astype(str).str.strip().tolist()
                tab = df.iloc[rid+1:].copy().reset_index(drop=True)
                tab.columns = header
                return tab
        return pd.DataFrame()
    start_idx = 0
    for i, it in enumerate(lower_items):
        if it.startswith("month"):
            start_idx = i
            break
    items = items[start_idx:]
    chunk = 5
    rows = []
    i = 0
    while i + chunk - 1 < len(items):
        block = items[i:i+chunk]
        rows.append(block)
        i += chunk
    if not rows:
        return pd.DataFrame()
    dfn = pd.DataFrame(rows, columns=["Month", "SKU", "Stock", "IMS", "Shipment"])
    return dfn

def normalize_input(df_raw):
    if df_raw.empty:
        return pd.DataFrame()
    if df_raw.shape[1] > 3:
        header_row = None
        for r in range(min(6, df_raw.shape[0])):
            rowvals = df_raw.iloc[r].astype(str).str.strip().str.lower().tolist()
            if any("month" == v or "month" in v for v in rowvals) and any("sku" == v or "sku" in v for v in rowvals):
                header_row = r
                break
        if header_row is not None:
            header = df_raw.iloc[header_row].astype(str).str.strip().tolist()
            df_tab = df_raw.iloc[header_row+1:].copy().reset_index(drop=True)
            df_tab.columns = header
        else:
            df_tab = try_transpose_vertical_block(df_raw)
            if df_tab.empty:
                df_tab = df_raw.copy()
                df_tab.columns = [str(c).strip() for c in df_tab.iloc[0].astype(str).tolist()]
                df_tab = df_tab.iloc[1:].reset_index(drop=True)
    else:
        df_tab = try_transpose_vertical_block(df_raw)
        if df_tab.empty:
            df_tab = df_raw.copy()
            df_tab.columns = [str(c).strip() for c in df_tab.iloc[0].astype(str).tolist()]
            df_tab = df_tab.iloc[1:].reset_index(drop=True)

    df_tab.columns = [str(c).strip() for c in df_tab.columns]
    cols_map = {c.lower(): c for c in df_tab.columns}
    def find_col(possible):
        for p in possible:
            if p in cols_map:
                return cols_map[p]
        return None

    month_col = find_col(["month", "date", "period"])
    sku_col = find_col(["sku", "product", "item", "code", "description"])
    stock_col = find_col(["stock", "onhand", "inventory"])
    ims_col = find_col(["ims", "on invoice", "onmarket", "available"])
    ship_col = find_col(["shipment", "shipments", "shipped", "sales"])

    if month_col is None and set(map(str.lower, df_tab.columns)).issubset({str(i) for i in range(df_tab.shape[1])}):
        if df_tab.shape[1] >= 5:
            df_tab.columns = ["Month", "SKU", "Stock", "IMS", "Shipment"] + list(df_tab.columns[5:])
            month_col, sku_col, stock_col, ims_col, ship_col = "Month", "SKU", "Stock", "IMS", "Shipment"

    rename_map = {}
    if month_col:
        rename_map[month_col] = "Month"
    if sku_col:
        rename_map[sku_col] = "SKU"
    if stock_col:
        rename_map[stock_col] = "Stock"
    if ims_col:
        rename_map[ims_col] = "IMS"
    if ship_col:
        rename_map[ship_col] = "Shipment"

    df_tab = df_tab.rename(columns=rename_map)

    if "SKU" not in df_tab.columns:
        candidate = None
        for c in df_tab.columns:
            s = df_tab[c].astype(str).str.strip()
            nonnum_pct = (~s.str.match(r'^[\d\.\-\/]+$')).mean()
            uniq = s.nunique(dropna=True)
            if nonnum_pct > 0.5 and uniq < len(s) * 0.95:
                candidate = c
                break
        if candidate:
            df_tab = df_tab.rename(columns={candidate: "SKU"})
    if "SKU" not in df_tab.columns:
        df_tab["SKU"] = "SKU_1"

    df_tab["SKU"] = df_tab["SKU"].replace("", np.nan).ffill().fillna("SKU_1")

    if "Month" in df_tab.columns:
        df_tab["Month"] = pd.to_datetime(df_tab["Month"], errors="coerce")
        if df_tab["Month"].notna().any():
            df_tab["Month"] = df_tab["Month"].dt.to_period("M").dt.to_timestamp()

    for c in ["Stock", "IMS", "Shipment"]:
        if c in df_tab.columns:
            df_tab[c] = df_tab[c].astype(str).str.replace(",", "").str.replace(" ", "")
            df_tab[c] = pd.to_numeric(df_tab[c].replace("", "0"), errors="coerce").fillna(0)
        else:
            df_tab[c] = 0

    keep = ["Month", "SKU", "Stock", "IMS", "Shipment"]
    df_out = df_tab[[c for c in keep if c in df_tab.columns]].copy()
    df_out = df_out[~((df_out.get("Month").isna()) & (df_out[["Stock", "IMS", "Shipment"]].sum(axis=1) == 0))]
    df_out = df_out.reset_index(drop=True)

    st.sidebar.write("Parsed sample (first 12 rows):")
    st.sidebar.dataframe(df_out.head(12))
    st.sidebar.write("Detected columns:", list(df_out.columns))

    return df_out

def compute_forecast_per_sku(sku_df, months=12):
    sku_df = sku_df.sort_values("Month").set_index("Month").copy()
    if len(sku_df.index) >= 2:
        idx = pd.date_range(sku_df.index.min(), sku_df.index.max(), freq="MS")
        sku_df = sku_df.reindex(idx)
    for c in ["IMS", "Stock", "Shipment"]:
        if c in sku_df.columns:
            sku_df[c] = pd.to_numeric(sku_df[c], errors="coerce")
            sku_df[c] = sku_df[c].fillna(method="ffill").fillna(0)
    sku_df["Stock_Change"] = sku_df["Stock"].diff().fillna(0)
    sku_df["Net_Consumption"] = sku_df["IMS"].fillna(0) + sku_df["Stock_Change"].fillna(0)
    sku_df["Net_Consumption"] = sku_df["Net_Consumption"].replace([np.inf, -np.inf], 0).fillna(0)

    seasonal = np.zeros(months)
    try:
        if len(sku_df["Net_Consumption"].dropna()) >= 12:
            dec = seasonal_decompose(sku_df["Net_Consumption"], model="additive", period=12, extrapolate_trend="freq")
            seasonal_pattern = dec.seasonal[-12:]
            if len(seasonal_pattern) >= 12:
                seasonal = seasonal_pattern.values[:months]
    except Exception:
        seasonal = np.zeros(months)

    X = np.arange(len(sku_df)).reshape(-1, 1)
    y = sku_df["Net_Consumption"].values
    if len(X) >= 2 and np.any(y != 0):
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(sku_df), len(sku_df) + months).reshape(-1, 1)
        trend = model.predict(future_X)
    else:
        trend = np.full(months, sku_df["Net_Consumption"].mean() if len(y) > 0 else 0)

    forecast = trend + seasonal
    forecast = np.maximum(forecast, 0)

    future_months = pd.date_range(start=sku_df.index.max() + pd.DateOffset(months=1), periods=months, freq="MS")
    out = pd.DataFrame({
        "Month": future_months,
        "SKU": sku_df["SKU"].iloc[0] if "SKU" in sku_df.columns else "SKU_1",
        "Forecasted_Shipment": forecast
    })
    return out, sku_df

def highlight_alerts(forecast_df, last_ims):
    alerts = []
    for idx, row in forecast_df.iterrows():
        sku = row["SKU"]
        ims = last_ims.get(sku, 0)
        thresh = ims * (alert_threshold_percent / 100.0)
        if row["Forecasted_Shipment"] > thresh:
            alerts.append((sku, row["Month"].strftime("%Y-%m"), row["Forecasted_Shipment"], ims))
    return alerts

if uploaded is None:
    st.info("Upload your Excel file in the left panel. The app will try to auto-parse vertical or tabular layouts.")
    st.stop()

raw_bytes = uploaded.read()
df_raw = read_excel_bytes(io.BytesIO(raw_bytes))
df = normalize_input(df_raw)

if df.empty or "SKU" not in df.columns:
    st.error("Uploaded file could not be parsed into Month, SKU, Stock, IMS, Shipment. Check your file or try again.")
    st.stop()

st.subheader("Preview data (first 10 rows)")
st.dataframe(df.head(10))

if "SKU" not in df.columns:
    st.error("No SKU column found. Detected columns: " + ", ".join(list(df.columns)))
    st.stop()

skus = df["SKU"].unique()
st.sidebar.subheader("Select SKU for detailed view")
sel_sku = st.sidebar.selectbox("SKU", options=skus)

all_forecasts = []
last_ims = {}
sku_histories = {}
for sku in skus:
    sku_df = df[df["SKU"] == sku].copy()
    for c in ["Shipment", "IMS", "Stock"]:
        if c not in sku_df.columns:
            sku_df[c] = 0
    fc, hist = compute_forecast_per_sku(sku_df, months=months_to_forecast)
    all_forecasts.append(fc)
    sku_histories[sku] = hist
    if len(hist["IMS"].dropna()) > 0:
        last_ims[sku] = hist["IMS"].dropna().iloc[-1]
    else:
        last_ims[sku] = 0

forecast_all = pd.concat(all_forecasts, ignore_index=True)

st.subheader("Forecasts")
with st.expander("Download / Manual override", expanded=True):
    st.markdown("You can edit forecasted shipment values for any SKU/month and click Apply Overrides.")
    forecast_for_edit = forecast_all.copy()
    forecast_for_edit["Month"] = pd.to_datetime(forecast_for_edit["Month"]).dt.strftime("%Y-%m")

    # Use data editor if available, otherwise provide CSV download/upload fallback
    edited = None
    if hasattr(st, "experimental_data_editor"):
        try:
            edited = st.experimental_data_editor(forecast_for_edit, num_rows="dynamic")
        except Exception:
            edited = None

    if edited is None:
        st.info("Interactive editor not available in this Streamlit runtime. Use the CSV fallback below to apply manual overrides.")
        csv = forecast_for_edit.to_csv(index=False).encode("utf-8")
        st.download_button("Download editable CSV", data=csv, file_name="forecast_edit.csv", mime="text/csv")
        uploaded_csv = st.file_uploader("Upload edited CSV to apply overrides", type=["csv"])
        if uploaded_csv is not None:
            try:
                edited = pd.read_csv(uploaded_csv)
            except Exception:
                st.error("Could not read uploaded CSV. Ensure it's a valid CSV with columns: Month, SKU, Forecasted_Shipment")

    if edited is not None and st.button("Apply Overrides"):
        edited2 = edited.copy()
        edited2["Month"] = pd.to_datetime(edited2["Month"].astype(str) + "-01", errors="coerce")
        overrides = {}
        for _, r in edited2.iterrows():
            try:
                overrides[(r["SKU"], r["Month"].to_pydatetime())] = float(r["Forecasted_Shipment"])
            except Exception:
                pass
        def apply_over(row):
            key = (row["SKU"], pd.to_datetime(row["Month"]))
            return overrides.get(key, row["Forecasted_Shipment"])
        forecast_all["Month"] = pd.to_datetime(forecast_all["Month"])
        forecast_all["Forecasted_Shipment"] = forecast_all.apply(apply_over, axis=1)
        st.success("Overrides applied.")

st.dataframe(forecast_all.sort_values(["SKU", "Month"]).reset_index(drop=True))

def to_excel_bytes(df_out):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Forecast")
    return buf.getvalue()

excel_bytes = to_excel_bytes(forecast_all)
b64 = base64.b64encode(excel_bytes).decode()
href = f'<a href="data:application/octet-stream;base64,{b64}" download="shipment_forecast.xlsx">Download forecast as Excel</a>'
st.markdown(href, unsafe_allow_html=True)

alerts = highlight_alerts(forecast_all, last_ims)
st.subheader("Alerts")
if alerts:
    for a in alerts:
        sku, month, fval, ims = a
        st.warning(f"SKU {sku} | {month} → Forecast {fval:.1f} > {alert_threshold_percent}% of last IMS ({ims:.1f})")
else:
    st.success("No alerts")

st.subheader(f"SKU details — {sel_sku}")
hist = sku_histories[sel_sku]
hist = hist.reset_index().rename(columns={"index": "Month"})
hist["Month"] = pd.to_datetime(hist["Month"])
fc_sku = forecast_all[forecast_all["SKU"] == sel_sku].copy()
fc_sku["Month"] = pd.to_datetime(fc_sku["Month"])

fig = go.Figure()
if "IMS" in hist.columns:
    fig.add_trace(go.Bar(x=hist["Month"], y=hist["IMS"], name="IMS", marker_color="rgba(31,119,180,0.6)"))
if "Shipment" in hist.columns:
    fig.add_trace(go.Scatter(x=hist["Month"], y=hist["Shipment"], name="Actual Shipment", mode="lines+markers"))
fig.add_trace(go.Scatter(x=fc_sku["Month"], y=fc_sku["Forecasted_Shipment"], name="Forecasted Shipment", mode="lines+markers", line=dict(dash="dash")))
fig.update_layout(title=f"History + Forecast for {sel_sku}", xaxis_title="Month", yaxis_title="Units", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.info("Notes: Forecast uses simple linear trend + additive seasonality where available. Overrides update forecast values only.")
