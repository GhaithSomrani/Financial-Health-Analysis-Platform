# Corporate Financial Health Analyzer

A data science platform that evaluates the financial health of publicly traded companies by combining SEC filings, market data, and macroeconomic indicators — producing financial ratios, distress scores, and predictive risk assessments.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Foundational Knowledge](#2-foundational-knowledge)
3. [Data Sources](#3-data-sources)
4. [Project Architecture](#4-project-architecture)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Target Variables](#6-target-variables)
7. [Financial Metrics Reference](#7-financial-metrics-reference)
8. [Project Structure](#8-project-structure)
9. [Setup & Installation](#9-setup--installation)
10. [Usage](#10-usage)

---

## 1. Project Overview

This project answers a fundamental question in finance: **Is this company financially healthy, and how likely is it to face distress?**

It does this by:
- Pulling structured financial data from three authoritative sources (SEC, Yahoo Finance, World Bank)
- Computing a standardized set of financial ratios and metrics
- Calculating the Altman Z-Score as a traditional distress signal
- Training machine learning models to predict financial distress probability
- Visualizing company performance, peer comparisons, and risk trends

The result is a reproducible, data-driven framework for assessing corporate health — useful for investment screening, credit analysis, or academic research.

---

## 2. Foundational Knowledge

### 2.1 The Three Core Financial Statements

Every public company files three financial statements annually. These are the raw inputs for almost every metric in this project:

| Statement | What It Tells You | Key Lines |
|---|---|---|
| **Balance Sheet** | What a company owns vs. owes at a point in time | Total Assets, Total Liabilities, Shareholders' Equity |
| **Income Statement** | Revenue generated and costs incurred over a period | Revenue, Operating Income, Net Income, EBITDA |
| **Cash Flow Statement** | Actual cash moving in and out | Operating CF, Investing CF, Financing CF, Free Cash Flow |

> **Why all three matter:** Net income (from the Income Statement) can be manipulated through accounting choices. The Cash Flow Statement shows real cash — harder to fake. The Balance Sheet shows structural risk through debt levels.

### 2.2 Financial Ratios — Turning Numbers into Signals

Raw financial figures (e.g., "$5B in debt") are meaningless without context. Ratios normalize data so you can compare companies across sizes, industries, and time periods.

**Ratio categories and what each signals:**

**Liquidity** — Can the company meet short-term obligations?
- Current Ratio = Current Assets / Current Liabilities (>1 = generally safe)
- Quick Ratio = (Current Assets − Inventory) / Current Liabilities (stricter test)
- Cash Ratio = Cash / Current Liabilities (most conservative)

**Leverage / Solvency** — How much debt is the company carrying?
- Debt-to-Equity = Total Debt / Shareholders' Equity
- Debt-to-Assets = Total Debt / Total Assets
- Interest Coverage = EBIT / Interest Expense (can they service their debt?)

**Profitability** — Is the company generating returns?
- Return on Assets (ROA) = Net Income / Total Assets
- Return on Equity (ROE) = Net Income / Shareholders' Equity
- Net Profit Margin = Net Income / Revenue
- EBITDA Margin = EBITDA / Revenue

**Efficiency** — How well does the company use its resources?
- Asset Turnover = Revenue / Total Assets
- Receivables Turnover = Revenue / Accounts Receivable
- Inventory Turnover = COGS / Inventory

**Market / Valuation** — What does the market think the company is worth?
- Price-to-Earnings (P/E) = Stock Price / EPS
- Price-to-Book (P/B) = Market Cap / Book Value
- Market Cap / Revenue

**Growth** — Is the company expanding?
- Revenue Growth YoY
- Earnings Growth YoY
- Free Cash Flow Growth

### 2.3 The Altman Z-Score

Developed by Edward Altman in 1968, the Z-Score is a multi-factor formula that predicts bankruptcy within two years. It combines five ratios into a single score:

```
Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
```

Where:
- **X1** = Working Capital / Total Assets  _(liquidity)_
- **X2** = Retained Earnings / Total Assets  _(cumulative profitability)_
- **X3** = EBIT / Total Assets  _(operating efficiency)_
- **X4** = Market Value of Equity / Total Liabilities  _(market leverage)_
- **X5** = Revenue / Total Assets  _(asset efficiency)_

**Interpretation:**
| Z-Score | Zone | Interpretation |
|---|---|---|
| > 2.99 | Safe | Low distress risk |
| 1.81 – 2.99 | Grey | Ambiguous, monitor closely |
| < 1.81 | Distress | High bankruptcy risk |

> Note: The original formula was for manufacturing firms. Modified versions (Z' and Z'') exist for private companies and non-manufacturers.

### 2.4 Financial Distress — What We're Predicting

Financial distress exists on a spectrum:
1. **Liquidity stress** — difficulty meeting short-term payments
2. **Solvency stress** — liabilities exceed assets
3. **Default** — missed debt payments
4. **Bankruptcy** — legal restructuring or liquidation

The ML prediction model in this project predicts the probability of reaching distress within a defined forward window (e.g., 2 years), using labeled historical data where outcomes are known.

### 2.5 Why Macroeconomic Context Matters

A company's financial ratios can deteriorate not because of internal failure, but because the macro environment worsens. For example:
- Rising **interest rates** increase debt servicing costs across all companies
- **GDP contraction** reduces revenue for cyclical industries
- **Inflation** erodes real margins

By including World Bank macro indicators, the model can distinguish company-specific risk from broad economic headwinds — improving prediction accuracy and interpretability.

### 2.6 EDGAR and 10-K Filings

The SEC requires all public U.S. companies to file a **10-K annually** — a comprehensive report including audited financial statements. EDGAR (Electronic Data Gathering, Analysis, and Retrieval) is the SEC's public database of these filings.

The SEC also exposes a structured JSON API (`data.sec.gov`) that provides financial facts in machine-readable format — making it possible to extract balance sheet and income statement line items without parsing PDF documents.

---

## 3. Data Sources

| Source | Data Provided | Access Method |
|---|---|---|
| **SEC EDGAR** | 10-K filings: balance sheets, income statements, cash flows | REST API (`data.sec.gov`) |
| **Yahoo Finance** | Stock prices, volumes, market cap, P/E ratios | `yfinance` Python library |
| **World Bank** | GDP growth, inflation, interest rates by country/year | `wbdata` / `pandas-datareader` |

### Data Dictionary (Key Fields)

| Field | Source | Used For |
|---|---|---|
| Total Assets | SEC | Denominator in ROA, Z-Score X1/X2/X3/X5 |
| Total Liabilities | SEC | Leverage ratios, Z-Score X4 |
| Shareholders' Equity | SEC | ROE, Debt-to-Equity |
| Revenue | SEC | Margins, Asset Turnover, Z-Score X5 |
| Net Income | SEC | ROA, ROE, Profit Margin |
| EBIT | SEC | Interest Coverage, Z-Score X3 |
| Operating Cash Flow | SEC | Free Cash Flow, distress signals |
| Current Assets / Liabilities | SEC | Liquidity ratios, Z-Score X1 |
| Retained Earnings | SEC | Z-Score X2 |
| Stock Price + Shares Outstanding | Yahoo Finance | Market Cap, Z-Score X4, P/E, P/B |
| GDP Growth Rate | World Bank | Macro context feature |
| Inflation Rate | World Bank | Macro context feature |
| Interest Rates | World Bank | Macro context feature |

---

## 4. Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  SEC EDGAR   │  │  Yahoo Finance   │  │   World Bank     │   │
│  │  10-K JSON   │  │  Stock Prices    │  │  Macro Indicators│   │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘   │
│         └──────────────────┬┘                      │             │
│                            ▼                        │             │
│                     Raw Data Store (CSV / DB)       │             │
└────────────────────────────────────────────────────┼─────────────┘
                             │                        │
┌────────────────────────────▼────────────────────────▼────────────┐
│                     PROCESSING LAYER                             │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  Feature Engineering                                      │   │
│  │  - Financial Ratio Calculation (20+ ratios)               │   │
│  │  - Altman Z-Score                                         │   │
│  │  - Time-series Aggregation (multi-year trends)            │   │
│  │  - Macro Feature Joining                                  │   │
│  └──────────────────────────┬────────────────────────────────┘   │
└─────────────────────────────┼────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                      MODELING LAYER                              │
│  ┌────────────────────┐   ┌────────────────────────────────┐     │
│  │  Altman Z-Score    │   │  ML Distress Prediction        │     │
│  │  (rule-based)      │   │  Logistic Regression           │     │
│  │                    │   │  Random Forest                 │     │
│  │                    │   │  Gradient Boosting (XGBoost)   │     │
│  └────────────────────┘   └────────────────────────────────┘     │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                   VISUALIZATION LAYER                            │
│  - Company performance dashboards                                │
│  - Risk score trends over time                                   │
│  - Peer comparison charts                                        │
│  - Sector heatmaps                                               │
│  - Macro overlay on financial trends                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Roadmap

### Phase 1 — Data Collection (Current)

**Goal:** Build a reliable pipeline to fetch and store raw data from all three sources.

**Steps:**
1. `dataCollection.py` — Fetch all SEC-registered company metadata (CIK, ticker, SIC, exchanges)
2. For each company, fetch `companyfacts` JSON from `data.sec.gov/api/xbrl/companyfacts/{CIK}.json`
3. Parse XBRL-tagged financial line items: extract balance sheet, income statement, and cash flow fields
4. Pull Yahoo Finance data: historical prices, market cap, and key ratios for each ticker
5. Pull World Bank indicators: GDP growth, inflation, interest rates per country per year
6. Store everything in a structured format (CSV files or SQLite database)

**Output:** A flat table with one row per company per year containing all raw financial fields.

**Challenges & Solutions:**
- SEC rate limits → handled with `AdaptiveRateLimiter` in `dataCollection.py`
- Missing fields (companies report different XBRL tags) → fallback tag mappings
- Ticker-CIK mismatch → cross-reference SEC company_tickers.json

---

### Phase 2 — Data Processing & Feature Engineering

**Goal:** Transform raw financials into analysis-ready ratios and metrics.

**Steps:**
1. Clean and validate raw data (handle nulls, outliers, restatements)
2. Standardize XBRL tag names to consistent field names across companies
3. Calculate all financial ratios (see Section 6)
4. Compute Altman Z-Score for each company-year pair
5. Calculate YoY growth rates for key metrics
6. Join macroeconomic indicators by fiscal year and country
7. Create multi-year lag features for trend detection

**Output:** A feature matrix: one row per (company, year), columns are all computed metrics.

---

### Phase 3 — Financial Distress Prediction

**Goal:** Build a model that predicts the probability of financial distress 1–2 years ahead.

**Steps:**
1. Define distress labels: companies that later filed for bankruptcy, defaulted, or were delisted for financial reasons
2. Source distress labels from public bankruptcy databases (UCLA-LoPucki, Compustat, or manual EDGAR research)
3. Split data: train on historical labeled examples, test on held-out periods
4. Train baseline model: Logistic Regression with ratio features
5. Train ensemble models: Random Forest, XGBoost
6. Evaluate with precision/recall/AUC-ROC (class imbalance is expected — distress is rare)
7. Analyze feature importance: which ratios matter most?
8. Calibrate probabilities so model output is interpretable as a true probability

**Output:** A trained model that takes a company's annual ratios and outputs distress probability (0–1).

---

### Phase 4 — Visualization & Reporting

**Goal:** Make findings interpretable to non-technical stakeholders.

**Steps:**
1. Per-company dashboard: ratio trends over time, Z-Score evolution, distress probability score
2. Peer comparison: plot company ratios against industry median (SIC-based grouping)
3. Sector heatmap: color-coded grid of Z-Scores or distress probabilities by sector
4. Macro overlay: show how interest rates or GDP changes correlate with sector-wide ratio changes
5. Export reports: PDF or HTML summary per company

**Tools:** `matplotlib`, `seaborn`, `plotly`, optionally `streamlit` for interactive dashboards

---

## 6. Target Variables

Defining the target variable is one of the most critical decisions in this project. The options below range from simple rule-based labels to continuous outputs — each with different data requirements and analytical goals.

---

### Option 1 — Binary Classification (Primary Approach)

Predict **Yes/No** for a distress event within a 1–2 year forward window:

| Target | Definition | Source |
|---|---|---|
| `bankruptcy` | Company filed Chapter 7 or Chapter 11 | EDGAR, UCLA-LoPucki Bankruptcy Research Database |
| `default` | Missed a debt payment or underwent debt restructuring | Bloomberg, Moody's, EDGAR 8-K filings |
| `delisted_financial` | Stock delisted due to financial failure (not M&A or voluntary) | NYSE/NASDAQ delisting notices |
| `distress_composite` | Any of the above, OR Z-Score drops below 1.81 in the next year | Derived from your own computed data |

**Best for:** Maximum data availability, interpretable output, standard academic benchmarks.

> **Recommended starting point:** Use `distress_composite` with the Z-Score threshold as a proxy. This generates labels automatically from SEC data alone — no external label source required.

---

### Option 2 — Multi-class Classification

Predict **which risk zone** the company falls into:

**Altman Z-Score zones (3 classes):**
```
0 = Safe     (Z > 2.99)   — low distress risk
1 = Grey     (1.81 ≤ Z ≤ 2.99) — ambiguous, monitor closely
2 = Distress (Z < 1.81)   — high bankruptcy risk
```

**Credit risk tier (3 classes):**
```
0 = Investment Grade       — strong financial position
1 = High Yield / Speculative — elevated risk
2 = Distressed / Near-Default — immediate concern
```

**Best for:** Granular risk segmentation and peer benchmarking across zones.

---

### Option 3 — Regression (Continuous Output)

Predict a **numeric score** rather than a class:

| Target | Definition | Why Useful |
|---|---|---|
| `z_score_next_year` | Predict next year's Altman Z-Score | Captures trajectory, not just threshold crossing |
| `distress_probability` | Calibrated probability from a trained classifier | Readable as "37% chance of distress in 2 years" |
| `future_stock_return` | 1-year forward stock return as market-implied health proxy | Reflects market's collective financial assessment |
| `credit_spread` | Bond yield minus risk-free rate | Direct measure of perceived default risk |

**Best for:** Ranking companies by risk severity, continuous monitoring dashboards.

---

### Recommended Strategy for This Project

Given the available data sources, the optimal approach is layered:

```
Primary target:    binary `distress_flag`
                   → 1 if next year's Z-Score < 1.81, else 0
                   → Fully computable from SEC data alone

Secondary target:  3-class `altman_zone`
                   → Safe / Grey / Distress

Regression output: `z_score_next_year`
                   → Predict the actual score, not just the zone
```

---

### Handling Class Imbalance

Real distress events are rare (~1–3% of companies per year), creating severe class imbalance. Strategies to handle this:

| Strategy | Description |
|---|---|
| **SMOTE** | Synthetically oversample the minority (distressed) class |
| **Class weighting** | Penalize missing a distress case more heavily during training |
| **Threshold tuning** | Lower the decision threshold (e.g., flag at 30% probability, not 50%) |
| **Z-Score proxy label** | Increases positive cases since the grey zone is much larger than actual bankruptcies |

---

### Decision Guide

```
Do you have access to real bankruptcy / default labels?
  ├── YES → Binary classification with real labels + class weighting
  └── NO  → Use Z-Score < 1.81 as proxy label (fully derived from SEC data)
              ├── Binary:      distressed (1) vs. not distressed (0)
              └── Multi-class: Safe / Grey / Distress zones
```

---

## 7. Financial Metrics Reference

### Liquidity
| Metric | Formula |
|---|---|
| Current Ratio | Current Assets / Current Liabilities |
| Quick Ratio | (Current Assets − Inventory) / Current Liabilities |
| Cash Ratio | Cash & Equivalents / Current Liabilities |

### Leverage
| Metric | Formula |
|---|---|
| Debt-to-Equity | Total Debt / Shareholders' Equity |
| Debt-to-Assets | Total Debt / Total Assets |
| Interest Coverage | EBIT / Interest Expense |
| Net Debt / EBITDA | (Total Debt − Cash) / EBITDA |

### Profitability
| Metric | Formula |
|---|---|
| ROA | Net Income / Total Assets |
| ROE | Net Income / Shareholders' Equity |
| ROIC | NOPAT / Invested Capital |
| Gross Margin | Gross Profit / Revenue |
| EBITDA Margin | EBITDA / Revenue |
| Net Profit Margin | Net Income / Revenue |

### Efficiency
| Metric | Formula |
|---|---|
| Asset Turnover | Revenue / Total Assets |
| Receivables Turnover | Revenue / Accounts Receivable |
| Inventory Turnover | COGS / Inventory |
| Days Sales Outstanding | 365 / Receivables Turnover |

### Market / Valuation
| Metric | Formula |
|---|---|
| P/E Ratio | Stock Price / EPS |
| P/B Ratio | Market Cap / Book Value of Equity |
| EV/EBITDA | Enterprise Value / EBITDA |
| Market Cap / Revenue | Market Capitalization / Revenue |

### Altman Z-Score Components
| Variable | Formula | Weight |
|---|---|---|
| X1 | Working Capital / Total Assets | 1.2 |
| X2 | Retained Earnings / Total Assets | 1.4 |
| X3 | EBIT / Total Assets | 3.3 |
| X4 | Market Value of Equity / Total Liabilities | 0.6 |
| X5 | Revenue / Total Assets | 1.0 |

---

## 8. Project Structure

```
Project Data Science/
├── README.md
├── dataCollection.py          # Phase 1: SEC company metadata fetcher
├── main.ipynb                 # Main analysis notebook
├── data/
│   ├── company_tickers.csv    # All SEC-registered companies (CIK, ticker, SIC)
│   ├── checkpoint.json        # Resume state for data collection
│   ├── fetch.log              # Data collection logs
│   ├── companyfacts/          # Raw XBRL financial facts per company
│   └── submissions.zip        # SEC submission metadata archive
└── venv/                      # Python virtual environment
```

**Planned additions:**
```
├── src/
│   ├── parsers/
│   │   ├── sec_parser.py      # Extract financials from XBRL companyfacts
│   │   ├── yahoo_fetcher.py   # Pull stock price and market data
│   │   └── worldbank.py       # Fetch macro indicators
│   ├── features/
│   │   ├── ratios.py          # Financial ratio calculations
│   │   ├── zscore.py          # Altman Z-Score implementation
│   │   └── growth.py          # YoY growth rate features
│   ├── models/
│   │   ├── distress_model.py  # ML training and evaluation pipeline
│   │   └── evaluate.py        # AUC-ROC, precision/recall, calibration
│   └── viz/
│       ├── dashboard.py       # Per-company trend charts
│       ├── heatmap.py         # Sector risk heatmaps
│       └── comparison.py      # Peer benchmarking charts
├── notebooks/
│   ├── 01_exploration.ipynb   # Initial data exploration
│   ├── 02_features.ipynb      # Feature engineering experiments
│   ├── 03_modeling.ipynb      # Model training and evaluation
│   └── 04_visualization.ipynb # Charts and dashboard prototypes
└── tests/
    ├── test_ratios.py
    └── test_zscore.py
```

---

## 9. Setup & Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Project Data Science"

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install requests pandas tqdm yfinance wbdata scikit-learn xgboost \
            matplotlib seaborn plotly jupyter

# Run data collection
python dataCollection.py

# Open main notebook
jupyter notebook main.ipynb
```

**Required Python version:** 3.10+

---

## 10. Usage

### Step 1: Collect Company Metadata
```bash
python dataCollection.py
# Outputs: data/company_tickers.csv with ~10,000 SEC-registered companies
```

### Step 2: Run the Analysis Notebook
```bash
jupyter notebook main.ipynb
```

### Step 3: Filter Companies of Interest
```python
import pandas as pd

df = pd.read_csv("data/company_tickers.csv")

# Filter by exchange and sector (SIC code)
tech = df[df["exchanges"].str.contains("NYSE|NASDAQ", na=False)]
```

---

## Key References

- Altman, E.I. (1968). *Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy.* Journal of Finance.
- SEC EDGAR API: https://www.sec.gov/developer
- XBRL US Financial Data: https://data.sec.gov/api/xbrl/
- World Bank Open Data: https://data.worldbank.org
