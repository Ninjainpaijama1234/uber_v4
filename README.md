# README.md

# 🚖 Uber NCR 2024 — Analytics & Decision Lab (Lean)

**Purpose.** A single-file Streamlit app for fast exploration of Uber NCR 2024 bookings. Optimized to run on Streamlit Cloud / Python 3.13 with a reduced dependency set and strict numeric hygiene to avoid `isfinite`/dtype errors.

---

## 🔧 What’s Inside

- **Executive Overview:** KPIs, demand/revenue time series, funnel, stats, top frequencies.
- **Completion & Cancellation:** Rates, reasons, and segment breakdowns with auto “Insight”.
- **Geo & Temporal (Category view):** Busiest pickup/drop, hour/day peaks, heat tables.
- **Operational Efficiency:** VTAT/CTAT distributions, correlations, threshold hotspots.
- **Financials:** Revenue, ARPR, payment×vehicle stacked bars, value vs distance.
- **Ratings & Satisfaction:** Distributions, correlations, low-rating risk segments.
- **Incomplete Rides:** Volumes, reasons, breakdowns.
- **ML Lab (Lean, no sklearn):**
  - **Classification:** GLM(Logit) for `will_complete` with time-aware split.
  - **Forecasting:** ARIMA(D) daily demand with CIs.
  - **Clustering:** NumPy K-Means + PCA scatter + persona table.
  - **Regression:** OLS for `Booking Value` + diagnostics.
- **Risk & Fraud:** Z-score anomaly flags.
- **Operations Simulator:** Simple elastic what-ifs for supply, incentives, pricing.
- **Reports & Exports:** Filtered data, predictions, clusters, forecast, HTML summary.

---

## 📦 Data

**Hard-coded path (no uploader):**

**Strict columns (headers must match exactly):**


- Dates are **day-first**. `Date` + `Time` → `timestamp`.
- Reasons: `"0"` treated as missing; standardized to **Title Case**.
- Canonical statuses: `Completed`, `Customer Cancelled`, `Driver Cancelled`, `No Driver Found`, `Incomplete`.
- Target: `will_complete` = 1 for `Completed`, else 0.
- Heavy cardinality columns (Pickup/Drop) are compressed to **Top 30** for ML.

---

## 🏃 Run Locally

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
