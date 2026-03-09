import IPython.core.formatters
import os
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

companydata = pd.read_csv("data/company_tickers_filtered.csv")
print(companydata.info())


# def saveHistory(ticker):
#     print(ticker)
#     dat = yf.Ticker(ticker)
#     history = pd.DataFrame(dat.history(period="max"))
#     history.to_csv(f"data/stockprice/{ticker}.csv")


# tickerlist = companydata["ticker"].tolist()
# with ThreadPoolExecutor(max_workers=10) as executor:
#     executor.map(saveHistory, tickerlist)

tickers = companydata["ticker"].tolist()
print(tickers)


def doesStockExist(row):
    ticker = row["ticker"]
    # does the the file exist in data/stockprice/{ticker}.csv
    if os.path.exists(f"data/stockprice/{ticker}.csv"):
        return True
    return False


companydata["stock_exist"] = companydata.apply(doesStockExist, axis=1)
print(companydata.info())
exist = companydata[companydata["stock_exist"] == False]

# drop missing stock price from the main file from data company
companydata = companydata[companydata["stock_exist"] == True]
print(companydata.info())
companydata.to_csv("data/company_tickers_filtered_with_stock_price.csv", index=False)
