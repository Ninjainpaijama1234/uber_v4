# app.py
# FINAL VERSION - Hardcoded Filename
from __future__ import annotations

import os
import io
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------#
# App Config
# ------------------------------#
st.set_page_config(
    page_title="Uber NCR 2024 — Analytics & Decision Lab",
    page_icon="🚖",
    layout="wide",
)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Hardcoded Filename ---
# The app will now always look for this specific file.
# IMPORTANT: If this file is too large, the app will run out of memory.
# Change this to "ncr_ride_bookings_sample.csv" to use the smaller file.
DATA_FILE = "ncr_ride_bookingsv350k.csv"

# Schema (strict headers)
SCHEMA = [
    "Date", "Time", "Booking ID", "Booking Status", "Customer ID", "Vehicle Type",
    "Pickup Location", "Drop Location", "Avg VTAT", "Avg CTAT",
    "Cancelled Rides by Customer", "Reason for cancelling by Customer",
    "Cancelled Rides by Driver", "Driver Cancellation Reason",
    "Incomplete Rides", "Incomplete Rides Reason",
    "Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating", "Payment Method"
]
COLORS = {
    "insight": "#5e60ce", "demand": "#1f77b4", "risk": "#e76f51",
    "finance": "#2a9d8f", "cx": "#9b5de5",
}

# ------------------------------#
# Helper & Processing Functions
# ------------------------------#
@st.cache_data
def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Loads, validates, and preprocesses the dataset from a given file path."""
    msgs = []
    
    df = pd.read_csv(file_path)
    missing_cols = [c for c in SCHEMA if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    
    bad_rows = df["timestamp"].isna().sum()
    if bad_rows > 0:
        msgs.append(f"⚠️ Dropped {bad_rows} rows with invalid Date/Time.")
        df.dropna(subset=["timestamp"], inplace=True)
        if df.empty: return df, msgs

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    
    time_bins = [0, 5, 12, 17, 21, 24]
    time_labels = ["Night (21–04)", "Morning (05–11)", "Afternoon (12–16)", "Evening (17–20)", "Night (21–04)"]
    df["time_bucket"] = pd.cut(df["hour"], bins=time_bins, labels=time_labels, right=False, ordered=True)

    rename_map = {
        "Avg VTAT": "avg_vtat", "Avg CTAT": "avg_ctat", "Cancelled Rides by Customer": "cancelled_by_customer",
        "Cancelled Rides by Driver": "cancelled_by_driver", "Incomplete Rides": "incomplete_rides",
        "Booking Value": "booking_value", "Ride Distance": "ride_distance", "Driver Ratings": "driver_ratings",
        "Customer Rating": "customer_rating", "Reason for cancelling by Customer": "reason_customer",
        "Driver Cancellation Reason": "reason_driver", "Incomplete Rides Reason": "reason_incomplete"
    }
    df.rename(columns=rename_map, inplace=True)

    for col in rename_map.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    def clean_text(s):
        s_str = s.astype(str).str.strip().str.lower()
        return s_str.replace(['0', 'na', 'nan', 'none', 'null'], np.nan).str.title()

    for col in ["reason_customer", "reason_driver", "reason_incomplete"]:
        df[col] = clean_text(df[col])

    raw_status = df["Booking Status"].astype(str).str.lower().str.strip()
    conditions = [
        raw_status == "completed",
        raw_status.str.contains("no driver found", na=False),
        raw_status.str.contains("incomplete", na=False) | df["incomplete_rides"].gt(0),
        raw_status.str.contains("customer", na=False) | df["cancelled_by_customer"].gt(0),
        raw_status.str.contains("driver", na=False) | df["cancelled_by_driver"].gt(0)
    ]
    choices = ["Completed", "No Driver Found", "Incomplete", "Customer Cancelled", "Driver Cancelled"]
    df["booking_status_canon"] = np.select(conditions, choices, default=df["Booking Status"])
    df["will_complete"] = (df["booking_status_canon"] == "Completed").astype(int)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    df["booking_status_canon"] = df["booking_status_canon"].astype("category")

    return df.sort_values("timestamp").reset_index(drop=True), msgs

def filter_block(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("📅 Global Filters")
    min_d, max_d = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
    start_d, end_d = st.sidebar.date_input("Date range", (min_d, max_d), min_d, max_d)
    
    f_vtype = st.sidebar.multiselect("Vehicle Type", sorted(df["Vehicle Type"].cat.categories))
    f_pm = st.sidebar.multiselect("Payment Method", sorted(df["Payment Method"].cat.categories))
    f_status = st.sidebar.multiselect("Booking Status", sorted(df["booking_status_canon"].cat.categories))
    
    df_f = df[(df["timestamp"].dt.date >= start_d) & (df["timestamp"].dt.date <= end_d)]
    if f_vtype: df_f = df_f[df_f["Vehicle Type"].isin(f_vtype)]
    if f_pm: df_f = df_f[df_f["Payment Method"].isin(f_pm)]
    if f_status: df_f = df_f[df_f["booking_status_canon"].isin(f_status)]
    return df_f

# ------------------------------#
# Main App Logic
# ------------------------------#
def main():
    st.sidebar.title("🚖 Uber NCR 2024 — Analytics & Decision Lab")
    
    df = None
    try:
        if os.path.exists(DATA_FILE):
            df, load_msgs = load_and_preprocess_data(DATA_FILE)
            for m in load_msgs:
                st.warning(m)
        else:
            st.error(f"Error: The data file '{DATA_FILE}' was not found.")
            st.info(f"Please make sure the file is in your GitHub repository and the name matches exactly.")
            return

    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
        return

    if df.empty:
        st.error("No data available to display. The file might be empty or all rows were removed during cleaning.")
        return

    df_f = filter_block(df)

    if df_f.empty:
        st.info("No rows match the current filters.")
        return

    st.sidebar.markdown("---")
    st.sidebar.download_button("Download Filtered Data (CSV)", df_f.to_csv(index=False).encode("utf-8"),
                               "filtered_data.csv", "text/csv")

    tabs = st.tabs([
        "1) Executive Overview", "2) Ride Completion & Cancellation", "3) Geographical & Temporal",
        "4) Operational Efficiency", "5) Financial Analysis", "6) Ratings & Satisfaction",
        "7) Incomplete Rides", "8) ML Lab", "9) Risk & Fraud",
        "10) Operations Simulator", "11) Reports & Exports",
    ])

    # ---------- Tab 1: Executive Overview
    with tabs[0]:
        st.markdown("## Executive Overview")
        total = len(df_f)
        comp = int((df_f["booking_status_canon"] == "Completed").sum())
        revenue = df_f.loc[df_f["will_complete"] == 1, "booking_value"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Bookings", f"{total:,}")
        c2.metric("Completion %", f"{(comp/total*100) if total else 0:.1f}%")
        c3.metric("Total Revenue", f"₹ {revenue:,.0f}")
        c4.metric("Avg. Booking Value", f"₹ {(revenue/comp) if comp > 0 else 0:,.2f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Bookings Over Time")
            daily_bookings = df_f.set_index('timestamp').resample('D')['Booking ID'].count()
            st.line_chart(daily_bookings, color=COLORS['demand'])
        with col2:
            st.markdown("#### Booking Status Breakdown")
            status_counts = df_f['booking_status_canon'].value_counts()
            st.plotly_chart(px.pie(status_counts, values=status_counts.values, names=status_counts.index, hole=0.3), use_container_width=True)

    # ---------- Tab 2: Ride Completion & Cancellation
    with tabs[1]:
        st.markdown("## Ride Completion & Cancellation Analysis")
        total = len(df_f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Customer Cancel %", f"{(df_f['booking_status_canon'] == 'Customer Cancelled').sum() / total * 100:.1f}%")
        c2.metric("Driver Cancel %", f"{(df_f['booking_status_canon'] == 'Driver Cancelled').sum() / total * 100:.1f}%")
        c3.metric("No Driver Found %", f"{(df_f['booking_status_canon'] == 'No Driver Found').sum() / total * 100:.1f}%")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top Customer Cancellation Reasons")
            cust_reasons = df_f['reason_customer'].value_counts().nlargest(10)
            st.bar_chart(cust_reasons, color=COLORS['risk'])
        with c2:
            st.markdown("#### Top Driver Cancellation Reasons")
            driver_reasons = df_f['reason_driver'].value_counts().nlargest(10)
            st.bar_chart(driver_reasons, color=COLORS['risk'])

    # ---------- Tab 3: Geographical & Temporal
    with tabs[2]:
        st.markdown("## Geographical & Temporal Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Busiest Pickup Locations")
            top_pickups = df_f['Pickup Location'].value_counts().nlargest(10)
            st.bar_chart(top_pickups, color=COLORS['demand'])
        with c2:
            st.markdown("#### Busiest Drop Locations")
            top_drops = df_f['Drop Location'].value_counts().nlargest(10)
            st.bar_chart(top_drops, color=COLORS['demand'])

        st.markdown("---")
        
        st.markdown("#### Demand Heatmap (Hour of Day vs. Day of Week)")
        heatmap_data = df_f.pivot_table(index='hour', columns='weekday', values='Booking ID', aggfunc='count', fill_value=0)
        heatmap_data.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        st.plotly_chart(px.imshow(heatmap_data, labels=dict(x="Day of Week", y="Hour of Day", color="Bookings"),
                                  x=heatmap_data.columns, y=heatmap_data.index, aspect="auto",
                                  color_continuous_scale=px.colors.sequential.Viridis), use_container_width=True)

    # ---------- Tab 4: Operational Efficiency
    with tabs[3]:
        st.markdown("## Operational Efficiency")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Avg. Vehicle-to-Arrival Time (VTAT)")
            st.plotly_chart(px.histogram(df_f, x="avg_vtat", nbins=40, title="VTAT Distribution"), use_container_width=True)
        with c2:
            st.markdown("#### Avg. Customer-to-Arrival Time (CTAT)")
            st.plotly_chart(px.histogram(df_f, x="avg_ctat", nbins=40, title="CTAT Distribution"), use_container_width=True)
        
    # ---------- Tab 5: Financial Analysis
    with tabs[4]:
        st.markdown("## Financial Analysis")
        completed_rides = df_f[df_f['booking_status_canon'] == 'Completed']
        st.plotly_chart(px.scatter(completed_rides, x='ride_distance', y='booking_value', 
                                   color='Vehicle Type', title='Booking Value vs. Ride Distance for Completed Rides',
                                   trendline='ols', trendline_scope='overall'), use_container_width=True)

    # ---------- Tab 6: Ratings & Satisfaction
    with tabs[5]:
        st.markdown("## Ratings & Customer Satisfaction")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Driver Ratings Distribution")
            st.plotly_chart(px.histogram(df_f, x="driver_ratings", nbins=20, title="Driver Ratings"), use_container_width=True)
        with c2:
            st.markdown("#### Customer Ratings Distribution")
            st.plotly_chart(px.histogram(df_f, x="customer_rating", nbins=20, title="Customer Ratings"), use_container_width=True)

    # ---------- Tab 7: Incomplete Rides
    with tabs[6]:
        st.markdown("## Incomplete Rides Analysis")
        incomplete_df = df_f[df_f['booking_status_canon'] == 'Incomplete']
        if not incomplete_df.empty:
            reason_counts = incomplete_df['reason_incomplete'].value_counts().nlargest(10)
            st.plotly_chart(px.bar(reason_counts, x=reason_counts.index, y=reason_counts.values, title="Top Reasons for Incomplete Rides"), use_container_width=True)
        else:
            st.info("No incomplete rides in the selected data.")

    # ---------- Tab 8: ML Lab (placeholder)
    with tabs[7]:
        st.markdown("## ML Lab (Under Construction)")
        st.info("Machine learning models for prediction and clustering will be available here.")

    # ---------- Tab 9: Risk & Fraud (placeholder)
    with tabs[8]:
        st.markdown("## Risk & Fraud (Under Construction)")
        st.info("Anomaly detection and risk scoring models will be available here.")

    # ---------- Tab 10: Operations Simulator (placeholder)
    with tabs[9]:
        st.markdown("## Operations Simulator (Under Construction)")
        st.info("A simulator for operational changes will be available here.")

    # ---------- Tab 11: Reports & Exports (placeholder)
    with tabs[10]:
        st.markdown("## Reports & Exports (Under Construction)")
        st.info("Custom report generation and data export functionality will be available here.")


if __name__ == "__main__":
    main()
