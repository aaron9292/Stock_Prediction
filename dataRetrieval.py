import yfinance as yf
import pandas as pd

def retrieveStockData(tickerSymbol, startDate, endDate, fileName):
    data_df = yf.download(tickerSymbol, start=startDate, end=endDate)['Adj Close']
    data_df.to_csv(fileName)

    f = pd.read_csv(fileName, index_col="Adj Close")
    f.drop(["Date"], axis=1, inplace=True)
    f.to_csv(fileName, header=False)