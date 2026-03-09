# SEC EDGAR Feature Extraction Guide

This guide explains how to extract every financial field required by this project from the SEC EDGAR `companyfacts` JSON API, using the XBRL tags defined in the `company_fact_schema`.

---

## 1. Schema Structure Overview

Each `companyfacts` JSON has the following top-level structure:

```json
{
  "cik": 320193,
  "entityName": "Apple Inc.",
  "facts": {
    "dei": { ... },
    "us-gaap": { ... }
  }
}
```

Every financial tag under `facts.us-gaap` (or `facts.dei`) follows this pattern:

```json
"Assets": {
  "label": "Assets",
  "description": "...",
  "units": {
    "USD": [
      {
        "start": "2022-10-01",   // period start (flow items only)
        "end":   "2023-09-30",   // period end / balance sheet date
        "val":   352755000000,   // the actual value
        "accn":  "0000320193-23-000106",
        "fy":    2023,           // fiscal year
        "fp":    "FY",           // fiscal period (FY / Q1 / Q2 / Q3)
        "form":  "10-K",         // SEC form type
        "filed": "2023-11-03",
        "frame": "CY2023Q3I"
      }
    ]
  }
}
```

**Key fields in each record:**

| Field | Meaning |
|---|---|
| `end` | Date the value applies to (balance sheet date or period end) |
| `val` | The numeric value (in USD unless noted) |
| `fy` | Fiscal year integer |
| `fp` | Fiscal period — use `"FY"` to get annual figures |
| `form` | SEC form — filter to `"10-K"` for annual reports |
| `start` | Present only on **flow** items (Income Statement, Cash Flow) |

> **Rule of thumb:** Filter `form == "10-K"` AND `fp == "FY"` to get one clean annual row per year.

---

## 2. Required XBRL Tags per Feature

### 2.1 Balance Sheet Fields

These are **point-in-time** items (no `start` date). Filter by `form == "10-K"` and `fp == "FY"`.

| Feature Needed | XBRL Tag(s) | Namespace | Unit |
|---|---|---|---|
| **Total Assets** | `Assets` | `us-gaap` | USD |
| **Current Assets** | `AssetsCurrent` | `us-gaap` | USD |
| **Total Liabilities** | `Liabilities` | `us-gaap` | USD |
| **Current Liabilities** | `LiabilitiesCurrent` | `us-gaap` | USD |
| **Shareholders' Equity** | `StockholdersEquity` | `us-gaap` | USD |
| **Retained Earnings** | `RetainedEarningsAccumulatedDeficit` | `us-gaap` | USD |
| **Inventory** | `InventoryNet` | `us-gaap` | USD |
| **Total Debt** | `LongTermDebt` → fallback `LongTermDebtNoncurrent` | `us-gaap` | USD |
| **Accounts Receivable** | `AccountsReceivableNetCurrent` | `us-gaap` | USD |

> **Working Capital** (needed for Altman X1) is derived, not a direct tag:
> `Working Capital = AssetsCurrent − LiabilitiesCurrent`

---

### 2.2 Income Statement Fields

These are **period** items (have both `start` and `end`). Filter for annual periods: `form == "10-K"` and `fp == "FY"`.

| Feature Needed | XBRL Tag(s) | Namespace | Unit | Notes |
|---|---|---|---|---|
| **Revenue** | `Revenues` | `us-gaap` | USD | Fallback: `RevenueFromContractWithCustomerExcludingAssessedTax` |
| **Gross Profit** | `GrossProfit` | `us-gaap` | USD | |
| **COGS** | `CostOfRevenue` | `us-gaap` | USD | |
| **Net Income** | `NetIncomeLoss` | `us-gaap` | USD | |
| **Operating Income / EBIT** | `OperatingIncomeLoss` | `us-gaap` | USD | Best EBIT proxy |
| **Interest Expense** | `InterestExpense` | `us-gaap` | USD | Fallback: `InterestExpenseNonoperating` |
| **EPS (Basic)** | `EarningsPerShareBasic` | `us-gaap` | USD/shares | |
| **EPS (Diluted)** | `EarningsPerShareDiluted` | `us-gaap` | USD/shares | |
| **D&A** | `DepreciationAndAmortization` | `us-gaap` | USD | Needed to compute EBITDA |

> **EBITDA** is derived: `EBITDA = OperatingIncomeLoss + DepreciationAndAmortization`

> **EBIT** is often directly equal to `OperatingIncomeLoss` for most companies. If needed more precisely: `EBIT = NetIncomeLoss + InterestExpense + IncomeTaxExpenseBenefit`

---

### 2.3 Cash Flow Statement Fields

Also **period** items. Filter `form == "10-K"` and `fp == "FY"`.

| Feature Needed | XBRL Tag | Namespace | Unit |
|---|---|---|---|
| **Operating Cash Flow** | `NetCashProvidedByUsedInOperatingActivities` | `us-gaap` | USD |
| **Investing Cash Flow** | `NetCashProvidedByUsedInInvestingActivities` | `us-gaap` | USD |
| **Financing Cash Flow** | `NetCashProvidedByUsedInFinancingActivities` | `us-gaap` | USD |
| **CapEx** | `PaymentsToAcquirePropertyPlantAndEquipment` | `us-gaap` | USD |

> **Free Cash Flow** is derived: `FCF = NetCashProvidedByUsedInOperatingActivities − PaymentsToAcquirePropertyPlantAndEquipment`

---

### 2.4 Shares Outstanding (from DEI namespace)

| Feature Needed | XBRL Tag | Namespace | Unit |
|---|---|---|---|
| **Shares Outstanding** | `EntityCommonStockSharesOutstanding` | `dei` | shares |

> This is used together with Yahoo Finance stock price to compute **Market Cap**:
> `Market Cap = Stock Price × EntityCommonStockSharesOutstanding`

---

## 3. Altman Z-Score Component Mapping

| Z-Score Variable | Formula | SEC Tags Needed |
|---|---|---|
| **X1** — Working Capital / Total Assets | `(AssetsCurrent − LiabilitiesCurrent) / Assets` | `AssetsCurrent`, `LiabilitiesCurrent`, `Assets` |
| **X2** — Retained Earnings / Total Assets | `RetainedEarningsAccumulatedDeficit / Assets` | `RetainedEarningsAccumulatedDeficit`, `Assets` |
| **X3** — EBIT / Total Assets | `OperatingIncomeLoss / Assets` | `OperatingIncomeLoss`, `Assets` |
| **X4** — Market Value of Equity / Total Liabilities | `(StockPrice × Shares) / Liabilities` | Yahoo Finance price + `EntityCommonStockSharesOutstanding`, `Liabilities` |
| **X5** — Revenue / Total Assets | `Revenues / Assets` | `Revenues`, `Assets` |

---

## 4. Python Extraction Pattern

### 4.1 Load a companyfacts JSON

```python
import json
import pandas as pd

def load_company_facts(cik: str) -> dict:
    path = f"data/companyfacts/CIK{cik.zfill(10)}.json"
    with open(path) as f:
        return json.load(f)
```

### 4.2 Generic tag extractor

```python
def extract_tag(facts: dict, tag: str, namespace: str = "us-gaap") -> pd.DataFrame:
    """
    Extract annual (10-K, FY) records for a given XBRL tag.
    Returns a DataFrame with columns: fy, end, val.
    """
    try:
        units = facts["facts"][namespace][tag]["units"]
    except KeyError:
        return pd.DataFrame()  # tag not reported by this company

    # Units key is usually "USD", "shares", or "USD/shares"
    unit_key = list(units.keys())[0]
    records = units[unit_key]

    df = pd.DataFrame(records)

    # Keep only annual 10-K filings
    df = df[(df["form"] == "10-K") & (df["fp"] == "FY")]

    # Drop duplicates — keep the most recently filed value per fiscal year
    df = df.sort_values("filed").drop_duplicates(subset=["fy"], keep="last")

    return df[["fy", "end", "val"]].rename(columns={"val": tag})
```

### 4.3 Extract all required features for one company

```python
TAGS = {
    # Balance Sheet
    "Assets":                              ("us-gaap", "total_assets"),
    "AssetsCurrent":                       ("us-gaap", "current_assets"),
    "Liabilities":                         ("us-gaap", "total_liabilities"),
    "LiabilitiesCurrent":                  ("us-gaap", "current_liabilities"),
    "StockholdersEquity":                  ("us-gaap", "shareholders_equity"),
    "RetainedEarningsAccumulatedDeficit":  ("us-gaap", "retained_earnings"),
    "InventoryNet":                        ("us-gaap", "inventory"),
    "LongTermDebt":                        ("us-gaap", "long_term_debt"),
    "AccountsReceivableNetCurrent":        ("us-gaap", "accounts_receivable"),
    # Income Statement
    "Revenues":                                    ("us-gaap", "revenue"),
    "GrossProfit":                                 ("us-gaap", "gross_profit"),
    "CostOfRevenue":                               ("us-gaap", "cogs"),
    "NetIncomeLoss":                               ("us-gaap", "net_income"),
    "OperatingIncomeLoss":                         ("us-gaap", "operating_income"),
    "InterestExpense":                             ("us-gaap", "interest_expense"),
    "DepreciationAndAmortization":                 ("us-gaap", "da"),
    "EarningsPerShareBasic":                       ("us-gaap", "eps_basic"),
    # Cash Flow
    "NetCashProvidedByUsedInOperatingActivities":  ("us-gaap", "cfo"),
    "NetCashProvidedByUsedInInvestingActivities":  ("us-gaap", "cfi"),
    "NetCashProvidedByUsedInFinancingActivities":  ("us-gaap", "cff"),
    "PaymentsToAcquirePropertyPlantAndEquipment":  ("us-gaap", "capex"),
    # DEI
    "EntityCommonStockSharesOutstanding":          ("dei", "shares_outstanding"),
}

def extract_all_features(facts: dict) -> pd.DataFrame:
    result = None
    for tag, (namespace, col_name) in TAGS.items():
        df = extract_tag(facts, tag, namespace)
        if df.empty:
            continue
        df = df.rename(columns={tag: col_name})[["fy", col_name]]
        if result is None:
            result = df
        else:
            result = result.merge(df, on="fy", how="outer")

    if result is None:
        return pd.DataFrame()

    # Derived features
    result["working_capital"] = result["current_assets"] - result["current_liabilities"]
    result["ebitda"]          = result.get("operating_income", 0) + result.get("da", 0)
    result["free_cash_flow"]  = result.get("cfo", 0) - result.get("capex", 0)

    return result.sort_values("fy")
```

---

## 5. Handling Missing Tags (Fallback Strategy)

Not every company reports every XBRL tag. Use this fallback order:

| Feature | Primary Tag | Fallback Tag |
|---|---|---|
| Revenue | `Revenues` | `RevenueFromContractWithCustomerExcludingAssessedTax` |
| Interest Expense | `InterestExpense` | `InterestExpenseNonoperating` |
| Long-Term Debt | `LongTermDebt` | `LongTermDebtNoncurrent` |
| D&A | `DepreciationAndAmortization` | `DepreciationDepletionAndAmortization` → `Depreciation` |

```python
def extract_with_fallback(facts: dict, tags: list[str], namespace: str = "us-gaap") -> pd.DataFrame:
    """Try each tag in order, return the first one found."""
    for tag in tags:
        df = extract_tag(facts, tag, namespace)
        if not df.empty:
            return df
    return pd.DataFrame()
```

---

## 6. Final Output Schema

After processing, each row in your feature matrix represents **one company × one fiscal year**:

| Column | Source | Used In |
|---|---|---|
| `cik` | SEC metadata | Join key |
| `fy` | XBRL `fy` field | Join key |
| `total_assets` | `Assets` | ROA, Z-Score X1/X2/X3/X5 |
| `current_assets` | `AssetsCurrent` | Current Ratio, Z-Score X1 |
| `total_liabilities` | `Liabilities` | Leverage ratios, Z-Score X4 |
| `current_liabilities` | `LiabilitiesCurrent` | Liquidity ratios, Z-Score X1 |
| `shareholders_equity` | `StockholdersEquity` | ROE, Debt-to-Equity |
| `retained_earnings` | `RetainedEarningsAccumulatedDeficit` | Z-Score X2 |
| `inventory` | `InventoryNet` | Quick Ratio, Inventory Turnover |
| `long_term_debt` | `LongTermDebt` | Debt ratios |
| `accounts_receivable` | `AccountsReceivableNetCurrent` | Receivables Turnover |
| `revenue` | `Revenues` | Margins, Asset Turnover, Z-Score X5 |
| `gross_profit` | `GrossProfit` | Gross Margin |
| `cogs` | `CostOfRevenue` | Inventory Turnover |
| `net_income` | `NetIncomeLoss` | ROA, ROE, Net Margin |
| `operating_income` | `OperatingIncomeLoss` | EBIT proxy, Z-Score X3 |
| `interest_expense` | `InterestExpense` | Interest Coverage |
| `da` | `DepreciationAndAmortization` | EBITDA calc |
| `eps_basic` | `EarningsPerShareBasic` | P/E ratio |
| `cfo` | `NetCashProvidedByUsedInOperatingActivities` | FCF, distress signals |
| `capex` | `PaymentsToAcquirePropertyPlantAndEquipment` | FCF |
| `shares_outstanding` | `EntityCommonStockSharesOutstanding` | Market Cap, Z-Score X4 |
| `working_capital` | Derived | Z-Score X1 |
| `ebitda` | Derived | EBITDA Margin, Net Debt/EBITDA |
| `free_cash_flow` | Derived | Growth features, distress signal |
