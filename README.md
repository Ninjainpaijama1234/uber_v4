# README.md

# 🚖 Uber NCR 2024 — Analytics & Decision Lab (Lean, Hard-coded File)

**Purpose.** A single-file Streamlit app for fast exploration of Uber NCR 2024 bookings. It’s optimized for Streamlit Cloud / Python 3.13 with a reduced dependency set and strict numeric hygiene to avoid dtype issues.  
**Data file is hard-coded** to: `ncr_ride_bookingsv350k` (CSV) located in your GitHub repo.

---

## 🔧 Features (Lean & Reliable)

- **Executive Overview:** KPIs, demand/revenue time series, funnel, stats, top frequencies.
- **Completion & Cancellation:** Rates, reasons, and segment breakdowns with auto “Insight”.
- **Geo & Temporal (Category view):** Busiest pickup/drop, hour/day peaks, heat tables.
- **Operational Efficiency:** VTAT/CTAT distributions, correlations, threshold hotspots.
- **Financials:** Revenue, ARPR, payment×vehicle stacked bars, value vs distance with trendline.
- **Ratings & Satisfaction:** Distributions, correlations, low-rating risk segments.
- **Incomplete Rides:** Volumes, reasons, breakdowns.
- **ML Lab (no sklearn):**
  - **Classification:** GLM(Logit) for `will_complete` with time-aware split.
  - **Forecasting:** ARIMA(D) daily demand with CIs.
  - **Clustering:** NumPy K-Means + PCA scatter + persona table.
  - **Regression:** OLS for `Booking Value` + diagnostics.
- **Risk & Fraud:** Z-score anomaly flags.
- **Ops Simulator:** Elastic what-ifs for supply, incentives, pricing.
- **Reports & Exports:** Filtered data, predictions, clusters, forecast, HTML summary.

---

## 📦 Data

**Hard-coded basename:** `ncr_ride_bookingsv350k`  
**What the app looks for (in this order):**
1. `./ncr_ride_bookingsv350k.csv`
2. `./ncr_ride_bookingsv350k`
3. `./data/ncr_ride_bookingsv350k.csv`
4. `./datasets/ncr_ride_bookingsv350k.csv`
5. `/mnt/data/ncr_ride_bookingsv350k.csv`
6. `/mount/src/uber_v2/ncr_ride_bookingsv350k.csv`
7. A shallow scan of the repo for a file named `ncr_ride_bookingsv350k.csv`

> If the file isn’t found, the app shows a clear error banner with the paths searched.

**Strict columns (headers must match exactly):**

- Dates are **day-first**. `Date` + `Time` → `timestamp`.
- `"0"` in reasons → missing; standardized to **Title Case**.
- Canonical statuses: `Completed`, `Customer Cancelled`, `Driver Cancelled`, `No Driver Found`, `Incomplete`.
- Target: `will_complete` = 1 for `Completed`, else 0.
- High-cardinality locations compressed to **Top 30** for ML stability.

---

## 🏃 Run

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
