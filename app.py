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
st.title("Shipment Forecast Agent — Monthly, Multi‑SKU, Seasonality, Overrides")

st.sidebar.header("Upload / Settings")
uploaded = st.sidebar.file_uploader("Upload Excel file (monthly data)", type=["xlsx", "xls"])
months_to_forecast = st.sidebar.number_input("Forecast months", min_value=1, max_value=24, value=12)
alert_threshold_percent = st.sidebar.number_input("Alert threshold vs IMS (%)", min_value=10, max_value=500, value=150)

def read_excel_bytes(bytes_io):
    # Read file (try single-sheet then all sheets)
    try:
        df = pd.read_excel(bytes_io)
    except Exception:
        xls = pd.ExcelFile(bytes_io)
        sheets = []
        for name in xls.sheet_names:
            s = pd.read_excel(xls, sheet_name=name)
            sheets.append(s)
        df = pd.concat(sheets, ignore_index=True)

    # Drop fully empty top rows and reset index
    df = df.dropna(how="all").reset_index(drop=True)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Detect strong vertical layout: many header keywords in first column
    first_col = df.columns[0]
    col0_vals = df[first_col].astype(str).str.strip().str.lower().tolist()
    header_keywords = {"month", "sku", "stock", "ims", "shipment", "shipments", "onhand", "inventory"}
    hits = sum(1 for v in col0_vals[:30] if any(k in v for k in header_keywords))

    if hits >= 3:
        # Parse vertical blocks: read non-empty values from first column and chunk by 5
        items = [x for x in df[first_col].astype(str).tolist() if str(x).strip() != ""]
        # Find first 'month' label position and start there
        start = 0
        for i, it in enumerate(items):
            if it.strip().lower().startswith("month"):
                start = i
                break
        items = items[start:]
        parsed = []
        i = 0
        while i + 4 < len(items):
            m = items[i].strip()
            sku = items[i + 1].strip()
            stock = items[i + 2].strip()
            ims = items[i + 3].strip()
            shipment = items[i + 4].strip()
            parsed.append({"Month": m, "SKU": sku, "Stock": stock, "IMS": ims, "Shipment": shipment})
            i += 5
        if len(parsed) >= 1:
            df_clean = pd.DataFrame(parsed)
            df_clean["Month"] = pd.to_datetime(df_clean["Month"], errors="coerce")
            for c in ["Stock", "IMS", "Shipment"]:
                # remove commas, empty strings then coerce numeric
                df_clean[c] = pd.to_numeric(df_clean[c].astype(str).str.replace(",", "").replace("", "0"), errors="coerce").fillna(0)
            return df_clean

    # If not vertical, assume normal tabular layout: return trimmed headers
    return df

def normalize_input(df):
    st.sidebar.write("Parsed sample (first 10 rows):")
st.sidebar.dataframe(df.head(10))
st.sidebar.write("Detected columns:", list(df.columns))
    # Drop completely empty top rows and reset index
    df = df.copy()
    # drop rows where all cells are NaN
    df = df.dropna(how="all").reset_index(drop=True)

    # Build lowercase->original mapping of column names
    cols = {str(c).strip().lower(): c for c in df.columns}

    # Attempt to match required fields case-insensitively and with common variants
    wanted = {
        "month": ["month", "date", "period"],
        "sku": ["sku", "product", "item", "code"],
        "stock": ["stock", "onhand", "inventory"],
        "ims": ["ims", "on invoice", "onmarket", "available"],
        "shipment": ["shipment", "shipments", "shipped", "sales"]
    }

    found = {}
    for key, variants in wanted.items():
        for v in variants:
            if v in cols:
                found[key] = cols[v]
                break

    # If Month found, convert to datetime
    if "month" in found:
        df[found["month"]] = pd.to_datetime(df[found["month"]], errors="coerce")

    # If SKU not found, create a default SKU column
    if "sku" not in found:
        df["SKU"] = "SKU_1"
        found["sku"] = "SKU"

    # Rename the detected columns to canonical names
    rename_map = {}
    for k, orig in found.items():
        rename_map[orig] = k.capitalize()

    df = df.rename(columns=rename_map)

    # Keep only relevant columns if present
    keep = [c for c in ["Month", "SKU", "Stock", "IMS", "Shipment"] if c in df.columns]
    result = df[keep].copy()

    # Debug: write detected columns to the Streamlit app so you can see what was found
    st.sidebar.write("Detected columns:", list(result.columns))

    return result

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
    st.info("Upload your Excel file in the left panel. Required (or auto-detected) columns: Month, SKU, Stock, IMS, Shipment.")
    st.stop()

raw_bytes = uploaded.read()
df_raw = read_excel_bytes(io.BytesIO(raw_bytes))
df = normalize_input(df_raw)

if df.empty:
    st.error("Uploaded file could not be parsed or contains no usable data.")
    st.stop()

st.subheader("Preview data (first 10 rows)")
st.dataframe(df.head(10))

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
    if len(hist["IMS"].dropna())>0:
        last_ims[sku] = hist["IMS"].dropna().iloc[-1]
    else:
        last_ims[sku] = 0

forecast_all = pd.concat(all_forecasts, ignore_index=True)

st.subheader("Forecasts")
with st.expander("Download / Manual override", expanded=True):
    st.markdown("You can edit forecasted shipment values for any SKU/month and click Apply Overrides.")
    forecast_for_edit = forecast_all.copy()
    forecast_for_edit["Month"] = pd.to_datetime(forecast_for_edit["Month"]).dt.strftime("%Y-%m")
    edited = st.experimental_data_editor(forecast_for_edit, num_rows="dynamic")
    if st.button("Apply Overrides"):
        edited2 = edited.copy()
        edited2["Month"] = pd.to_datetime(edited2["Month"] + "-01")
        overrides = {}
        for _, r in edited2.iterrows():
            overrides[(r["SKU"], r["Month"].to_pydatetime())] = float(r["Forecasted_Shipment"])
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
hist = hist.reset_index().rename(columns={"index":"Month"})
hist["Month"] = pd.to_datetime(hist["Month"])
fc_sku = forecast_all[forecast_all["SKU"]==sel_sku].copy()
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
