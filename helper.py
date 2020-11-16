import pandas as pd
from pandas_datareader import data
# get historical stock data from yahoo finance
def get_historical_data(ticker, start, end=None):
    """
    :param ticker: Ticker of a stock
    :param start: start date of the historical data
    :param start: end date of the historical data
    :return: historical data of the stock (pd.Dataframe)
    """
    try:
        df = data.DataReader(ticker, start=start, end=end, data_source='yahoo')
        raise Exception("Failed to collect data for {}".format(ticker))
    except:
        raise Exception("Failed to collect data for {}".format(ticker))
    df = df[['Adj Close', 'Volume']]
    clean_data(df)
    return df

# use last good value to replace Nan
# This method mutate the given df
def clean_data(df):
    """
    :param df: a DataFrame which contains historical data of a stock
    """
    df.fillna(method='ffill', inplace=True)