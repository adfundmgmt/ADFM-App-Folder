# 🧠 AD Fund Management LP – Internal Analytics Suite

This Streamlit-powered app hosts interactive tools built by AD Fund Management LP to uncover historical market patterns and support tactical equity decision making.

---

## 🔧 Tools Included

### 📈 Monthly Seasonality Explorer
Visualizes median monthly returns and hit rates for any equity, index, or commodity. Uses Yahoo Finance as the primary source, with automatic fallback to FRED for long-term S&P 500, Dow, and Nasdaq data.

- Green bars = months with positive median returns  
- Red bars = months with negative median returns  
- Black diamonds = frequency of positive returns ("hit rate")  

**Use Case**: Identify seasonal patterns (e.g. "Sell in May", "Santa Rally") and inform timing decisions.

---

### 📘 YTD Analog Year Correlation Explorer
Compares the current year-to-date return trajectory of any index or stock to historical years, highlighting the most statistically similar periods based on daily return path correlations.

- Fully interactive chart with year overlays  
- Adjustable correlation windows and filters  
- Designed to spark forward-looking hypothesis generation  

**Use Case**: Spot “analog years” and study how similar regimes unfolded.

---
