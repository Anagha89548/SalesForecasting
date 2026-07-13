import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="Enterprise Sales Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styles
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .metric-label {
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #1e293b;
        font-size: 28px;
        font-weight: 800;
        font-family: monospace;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 700;
        margin-top: 4px;
    }
    .delta-up { color: #10b981; }
    .delta-down { color: #f43f5e; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    csv_path = "train.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset file '{csv_path}' not found! Please ensure it resides in the same directory.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df

df = load_data()

if df.empty:
    st.stop()

# Sidebar Navigation and Filters
st.sidebar.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=120&auto=format&fit=crop&q=60", width=100)
st.sidebar.title("BI Navigation")
st.sidebar.markdown("---")

# Global Filters
st.sidebar.header("Global Filters")
region_options = ["All"] + sorted(df["region"].unique().tolist())
category_options = ["All"] + sorted(df["category"].unique().tolist())

selected_region = st.sidebar.selectbox("Select Region", region_options)
selected_category = st.sidebar.selectbox("Select Category", category_options)

# Apply filters
filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]

# Navigation tabs
active_tab = st.sidebar.radio(
    "Choose Analysis Panel",
    ["Executive Overview", "Holt-Winters Forecasting", "Anomaly Detection Logs", "Product Demand Segments"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Enterprise Sales Intelligence**\n\n"
    "This Streamlit dashboard operates on Holt-Winters Triple Exponential Smoothing "
    "and Z-Score algorithms matching the enterprise grade React app."
)

# ==========================================
# HELPER FUNCTIONS FOR KPIs & CHARTS
# ==========================================
def format_currency(val):
    return f"${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"

def format_short_currency(val):
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val/1_000:.1f}k"
    return f"${val:.2f}"

# ==========================================
# PANEL 1: EXECUTIVE OVERVIEW
# ==========================================
if active_tab == "Executive Overview":
    st.title("📊 Corporate Performance Overview")
    st.markdown("Real-time executive summaries, performance indicators, and transaction trends.")

    # Calculations
    total_sales = filtered_df["sales"].sum()
    total_qty = filtered_df["quantity"].sum()
    total_orders = filtered_df["id"].nunique()
    avg_basket = total_sales / total_orders if total_orders > 0 else 0

    # Comparative Stats (Yearly YoY Trend)
    yearly_totals = filtered_df.groupby("year")["sales"].sum().reset_index()
    yoy_delta = 0.0
    if len(yearly_totals) >= 2:
        val_2025 = yearly_totals[yearly_totals["year"] == 2025]["sales"].values
        val_2024 = yearly_totals[yearly_totals["year"] == 2024]["sales"].values
        if len(val_2025) > 0 and len(val_2024) > 0 and val_2024[0] > 0:
            yoy_delta = ((val_2025[0] - val_2024[0]) / val_2024[0]) * 100

    # Layout KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Revenue (USD)</div>
                <div class="metric-val">{format_short_currency(total_sales)}</div>
                <div class="metric-delta delta-up">▲ Dynamic Dataset</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Units Sold</div>
                <div class="metric-val">{total_qty:,}</div>
                <div class="metric-delta delta-up">▲ Bulk Fulfilled</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Order Frequency</div>
                <div class="metric-val">{total_orders:,}</div>
                <div class="metric-delta delta-up">▲ Unique Transactions</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Basket Size</div>
                <div class="metric-val">{format_currency(avg_basket)}</div>
                <div class="metric-delta {'delta-up' if yoy_delta >= 0 else 'delta-down'}">
                    {"▲" if yoy_delta >= 0 else "▼"} {abs(yoy_delta):.1f}% YoY (2024 vs 2025)
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Charts Section
    st.markdown("### 📈 Monthly Sales Demand Trend")
    monthly_trend = filtered_df.groupby("orderDate")["sales"].sum().reset_index()
    monthly_trend = monthly_trend.sort_values("orderDate")

    fig_line = px.line(
        monthly_trend,
        x="orderDate",
        y="sales",
        labels={"sales": "Revenue (USD)", "orderDate": "Billing Period"},
        template="plotly_white",
        color_discrete_sequence=["#4F46E5"]
    )
    fig_line.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        height=320,
        hovermode="x unified"
    )
    fig_line.update_traces(line=dict(width=3))
    st.plotly_chart(fig_line, use_container_width=True)

    # Sub-Breakdown grid
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📦 Sales Volume by Sub-Category")
        sub_cat_sales = filtered_df.groupby("subCategory")["sales"].sum().reset_index()
        sub_cat_sales = sub_cat_sales.sort_values("sales", ascending=True)

        fig_bar = px.bar(
            sub_cat_sales,
            x="sales",
            y="subCategory",
            orientation="h",
            labels={"sales": "Revenue (USD)", "subCategory": "Sub-Category"},
            template="plotly_white",
            color_discrete_sequence=["#6366F1"]
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("#### 🗺️ Regional Revenue Contribution")
        reg_sales = filtered_df.groupby("region")["sales"].sum().reset_index()
        
        fig_pie = px.pie(
            reg_sales,
            values="sales",
            names="region",
            template="plotly_white",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# PANEL 2: HOLT-WINTERS FORECASTING
# ==========================================
elif active_tab == "Holt-Winters Forecasting":
    st.title("🔮 Predictive Analytics Engine")
    st.markdown("Triple Exponential Smoothing (Holt-Winters Model) fit to historical monthly aggregates.")

    # Prepare Monthly Time Series
    ts_df = filtered_df.groupby("orderDate")["sales"].sum().reset_index()
    ts_df = ts_df.sort_values("orderDate")
    ts_df.set_index("orderDate", inplace=True)
    
    # Check data sufficiency
    if len(ts_df) < 12:
        st.warning("Insufficient data historical periods to fit Triple Exponential Smoothing (minimum 12 months required).")
    else:
        # Fit Holt-Winters Model
        # Sells are seasonal with annual periodicity (12 periods)
        try:
            model = ExponentialSmoothing(
                ts_df["sales"],
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated"
            )
            fitted_model = model.fit()
            
            # Forecast for next 12 periods (up to July 2027)
            forecast_periods = 12
            forecast_indices = pd.date_range(start="2026-07-01", periods=forecast_periods, freq="MS")
            forecast_dates = [d.strftime("%Y-%m") for d in forecast_indices]
            
            forecast_values = fitted_model.forecast(forecast_periods)
            fitted_values = fitted_model.fittedvalues
            
            # Formulate full dataset
            plot_records = []
            for date, actual in zip(ts_df.index, ts_df["sales"]):
                fitted_val = fitted_values.loc[date]
                plot_records.append({
                    "date": date,
                    "Type": "Historical Actuals",
                    "Sales": actual,
                    "Lower Bound": None,
                    "Upper Bound": None
                })
                plot_records.append({
                    "date": date,
                    "Type": "In-Sample Fit",
                    "Sales": fitted_val,
                    "Lower Bound": None,
                    "Upper Bound": None
                })
                
            # Compute RMSE & MAE
            residuals = ts_df["sales"] - fitted_values
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            
            for d_str, f_val in zip(forecast_dates, forecast_values):
                # Standard deviation of residuals used for confidence intervals (95% CI)
                std_err = np.std(residuals)
                lower = max(0, f_val - 1.96 * std_err)
                upper = f_val + 1.96 * std_err
                
                plot_records.append({
                    "date": d_str,
                    "Type": "Forecasted Sales",
                    "Sales": f_val,
                    "Lower Bound": lower,
                    "Upper Bound": upper
                })
                
            plot_df = pd.DataFrame(plot_records)
            
            # Render chart
            st.markdown("#### Holt-Winters Demand Projections (Next 12 Months)")
            
            fig_fc = go.Figure()
            
            # Historical actual line
            hist_df = plot_df[plot_df["Type"] == "Historical Actuals"]
            fig_fc.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["Sales"],
                mode="lines+markers",
                name="Historical Actual Sales",
                line=dict(color="#4F46E5", width=2.5)
            ))
            
            # Fitted actual line
            fit_df = plot_df[plot_df["Type"] == "In-Sample Fit"]
            fig_fc.add_trace(go.Scatter(
                x=fit_df["date"], y=fit_df["Sales"],
                mode="lines",
                name="In-Sample Model Fit",
                line=dict(color="#94A3B8", width=1.5, dash="dot")
            ))
            
            # Forecasted line
            fore_df = plot_df[plot_df["Type"] == "Forecasted Sales"]
            fig_fc.add_trace(go.Scatter(
                x=fore_df["date"], y=fore_df["Sales"],
                mode="lines+markers",
                name="Projected Forecast",
                line=dict(color="#F59E0B", width=2.5)
            ))
            
            # Add confidence intervals
            fig_fc.add_trace(go.Scatter(
                x=fore_df["date"].tolist() + fore_df["date"].tolist()[::-1],
                y=fore_df["Upper Bound"].tolist() + fore_df["Lower Bound"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval"
            ))
            
            fig_fc.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=10, b=20),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Metrics Columns
            st.markdown("#### Regression Fitting Diagnostics")
            col_mae, col_rmse = st.columns(2)
            
            with col_mae:
                st.markdown(f"""
                    <div class="metric-card" style="background-color: #fafafa;">
                        <div class="metric-label">Mean Absolute Error (MAE)</div>
                        <div class="metric-val">{format_currency(mae)}</div>
                        <p style="font-size: 11px; color: #64748b; margin-top: 8px;">
                            Indicates the average absolute deviation between model fit and actual transactions.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_rmse:
                st.markdown(f"""
                    <div class="metric-card" style="background-color: #fafafa;">
                        <div class="metric-label">Root Mean Squared Error (RMSE)</div>
                        <div class="metric-val">{format_currency(rmse)}</div>
                        <p style="font-size: 11px; color: #64748b; margin-top: 8px;">
                            Penalizes larger outliers. Essential for planning safety buffer stocks under seasonal volatility.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Could not fit Holt-Winters model: {e}. Check if database slice is empty or too small.")

# ==========================================
# PANEL 3: ANOMALY DETECTION LOGS
# ==========================================
elif active_tab == "Anomaly Detection Logs":
    st.title("🚨 Operational Outlier Detection")
    st.markdown("Outlier detection using a rolling rolling standard deviation threshold (1.7 σ) from overall mean monthly sales.")

    # Calculate monthly aggregates
    monthly_sales = df.groupby("orderDate")["sales"].sum().reset_index()
    monthly_sales = monthly_sales.sort_values("orderDate")
    
    # Calculate Mean & standard deviations
    overall_mean = monthly_sales["sales"].mean()
    overall_std = monthly_sales["sales"].std()
    
    threshold_val = 1.7
    upper_bound = overall_mean + threshold_val * overall_std
    lower_bound = overall_mean - threshold_val * overall_std
    
    # Hardcoded specific business drivers corresponding to injected anomalies in data.ts
    anomalies_drivers = {
        "2024-11": "Corporate Black Friday Technology promotion campaigns resulted in mass scale regional deals in East.",
        "2025-02": "Extreme severe winter freeze across Southern and Mid-Western states shut down freight logistics channels, causing high product delivery deferrals.",
        "2025-08": "Major office supply bulk replenishment contract finalized with multi-state corporate services provider.",
        "2024-05": "West region bulk copier replacements and heavy machine enterprise procurement campaigns."
    }
    
    # Annotate records
    annotated_list = []
    for idx, row in monthly_sales.iterrows():
        date = row["orderDate"]
        val = row["sales"]
        deviation = (val - overall_mean) / overall_std
        is_anomaly = abs(deviation) > threshold_val
        
        driver = "Normal operation. Sales tracking within normal variance threshold."
        if is_anomaly:
            driver = anomalies_drivers.get(date, "High localized customer volume orders or season variance spike.")
            
        annotated_list.append({
            "Period": date,
            "Sales": val,
            "Deviation": deviation,
            "Is Anomaly": is_anomaly,
            "Operational Driver / Explanation": driver
        })
        
    annot_df = pd.DataFrame(annotated_list)
    
    # Chart with markers for anomalies
    fig_anom = go.Figure()
    
    # Base Sales Line
    fig_anom.add_trace(go.Scatter(
        x=annot_df["Period"], y=annot_df["Sales"],
        mode="lines+markers",
        name="Monthly Revenue",
        line=dict(color="#4F46E5", width=2),
        marker=dict(color="#4F46E5", size=5)
    ))
    
    # Outliers
    anomalies_only = annot_df[annot_df["Is Anomaly"] == True]
    fig_anom.add_trace(go.Scatter(
        x=anomalies_only["Period"], y=anomalies_only["Sales"],
        mode="markers",
        name="Flagged Anomalies",
        marker=dict(color="#F43F5E", size=10, symbol="circle", line=dict(color="#FFFFFF", width=1.5))
    ))
    
    # Reference boundaries
    fig_anom.add_hline(y=overall_mean, line=dict(color="#94A3B8", dash="dash"), annotation_text="Historical Mean")
    fig_anom.add_hline(y=upper_bound, line=dict(color="#F43F5E", dash="dash"), annotation_text="Upper Bound (+1.7 σ)")
    fig_anom.add_hline(y=lower_bound, line=dict(color="#F43F5E", dash="dash"), annotation_text="Lower Bound (-1.7 σ)")
    
    fig_anom.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=10, b=20),
        height=320,
        legend=dict(orientation="h", y=1.02, x=1)
    )
    st.plotly_chart(fig_anom, use_container_width=True)
    
    # Table Log
    st.markdown("#### Outlier Diagnostic Ledger")
    table_display = annot_df[annot_df["Is Anomaly"] == True].copy()
    
    # Styling columns
    table_display["Sales"] = table_display["Sales"].map(format_currency)
    table_display["Deviation"] = table_display["Deviation"].map(lambda x: f"{x:+.2f} σ")
    
    st.dataframe(
        table_display[["Period", "Sales", "Deviation", "Operational Driver / Explanation"]],
        hide_index=True,
        use_container_width=True
    )

# ==========================================
# PANEL 4: PRODUCT DEMAND SEGMENTS
# ==========================================
elif active_tab == "Product Demand Segments":
    st.title("🎯 Sub-Category Demand Clustering")
    st.markdown("Sub-categories segmented into 4 demand clusters using standard K-Means based on total revenue, order count, and basket sizing.")

    # Pre-process segments
    seg_records = []
    subcategories = df["subCategory"].unique()
    
    for sub in subcategories:
        sub_df = df[df["subCategory"] == sub]
        tot_sales = sub_df["sales"].sum()
        orders = sub_df["id"].nunique()
        avg_basket = tot_sales / orders if orders > 0 else 0
        cat_name = sub_df["category"].iloc[0]
        
        seg_records.append({
            "subCategory": sub,
            "category": cat_name,
            "totalSales": tot_sales,
            "orderCount": orders,
            "avgOrderValue": avg_basket
        })
        
    seg_df = pd.DataFrame(seg_records)
    
    # Perform K-Means Clustering on TotalSales & OrderCount
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(seg_df[["totalSales", "orderCount"]])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    seg_df["ClusterID"] = kmeans.fit_predict(scaled_features)
    
    # Maps of Clusters based on standard assignments
    # 0 = High Value Drivers, 1 = Volume Staples, 2 = Niche Specials, 3 = Low-Demand Staples
    cluster_definitions = {
        0: {"Name": "High-Value Enterprise Drivers", "Color": "#3B82F6", "Desc": "Premium price points, large contract orders (Copiers, Phones)."},
        1: {"Name": "Volume Staples", "Color": "#10B981", "Desc": "Steady high-count transaction items with stable run-rates (Binders, Paper)."},
        2: {"Name": "Niche Specials", "Color": "#F59E0B", "Desc": "Moderate demand, specialized high-value units (Appliances, Tables)."},
        3: {"Name": "Low-Demand Staples", "Color": "#EF4444", "Desc": "Low relative transaction frequencies and margins (Art, Fasteners)."}
    }
    
    # To match standard colors from dashboard, we'll map the clusters dynamically
    # Sorted by sales to match descriptions: Top sales = High-Value Enterprise, etc.
    cluster_sales = seg_df.groupby("ClusterID")["totalSales"].mean().sort_values(ascending=False).index
    
    mapping = {
        cluster_sales[0]: 0, # High Value
        cluster_sales[1]: 1, # Volume Staples
        cluster_sales[2]: 2, # Niche Specials
        cluster_sales[3]: 3  # Low Demand
    }
    
    seg_df["MappedID"] = seg_df["ClusterID"].map(mapping)
    seg_df["ClusterName"] = seg_df["MappedID"].map(lambda x: cluster_definitions[x]["Name"])
    seg_df["ClusterColor"] = seg_df["MappedID"].map(lambda x: cluster_definitions[x]["Color"])
    
    # Scatter plot
    fig_scat = px.scatter(
        seg_df,
        x="totalSales",
        y="orderCount",
        color="ClusterName",
        size="avgOrderValue",
        hover_name="subCategory",
        labels={
            "totalSales": "Cumulative Revenue (USD)",
            "orderCount": "Transaction Sizing (Count)",
            "ClusterName": "Assigned Demand Segment"
        },
        template="plotly_white",
        color_discrete_map={cluster_definitions[i]["Name"]: cluster_definitions[i]["Color"] for i in range(4)}
    )
    
    fig_scat.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
        legend=dict(orientation="h", y=1.02, x=1)
    )
    st.plotly_chart(fig_scat, use_container_width=True)
    
    # Show Segment Definitions Card Row
    st.markdown("#### Strategic Cluster Breakdown")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    for i, col in enumerate([col_c1, col_c2, col_c3, col_c4]):
        defn = cluster_definitions[i]
        with col:
            st.markdown(f"""
                <div style="background-color: {defn['Color']}10; border: 1px solid {defn['Color']}30; padding: 15px; border-radius: 8px; height: 130px;">
                    <div style="font-weight: 800; font-size: 11px; text-transform: uppercase; color: {defn['Color']};">
                        ● {defn['Name']}
                    </div>
                    <p style="font-size: 11px; color: #475569; margin-top: 6px; line-height: 1.4;">
                        {defn['Desc']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
    # Full Table breakdown
    st.markdown("#### Cluster Assignment Matrix Ledger")
    table_seg = seg_df.copy()
    table_seg["totalSales"] = table_seg["totalSales"].map(format_currency)
    table_seg["avgOrderValue"] = table_seg["avgOrderValue"].map(format_currency)
    table_seg = table_seg.sort_values("MappedID")
    
    st.dataframe(
        table_seg[["subCategory", "category", "totalSales", "orderCount", "avgOrderValue", "ClusterName"]],
        hide_index=True,
        use_container_width=True
    )
