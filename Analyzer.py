import concurrent.futures
import os

from Stock import Stock
from scipy.stats import linregress
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class Analyzer:
    def __init__(self, tickers, start, end=None):
        """
        :param tickers: a list of ticker (str[])
        :param start: start date of the historical data (ex: '2010-01-01')
        :param end: end date of the historical data
        """
        self.stocks = {t:Stock(t) for t in tickers}
        self.start = start
        with concurrent.futures.ThreadPoolExecutor(os.cpu_count()+1) as executor:
            executor.map(lambda s: self.stocks[s].populate_df(start, end), self.stocks.keys())
        executor.shutdown(wait=True)

    def momentum(self, s, start_date, end_date, ranking_period):
        """
        :param s: stock
        :param start_date: datetime for start of ranking_period
        :param end_date: datetime for end of ranking_period
        :param ranking_period: length of ranking period (int)
        :return the momentum based on returns
        """
        returns = self.stocks[s].df['daily_return'].loc[start_date:end_date].dropna()
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        momentum = ((1 + slope) ** 252) * (rvalue ** 2) # annualize slope and multiply by R^2
        return  momentum
    
    # //TODO:
    # pick n stock that shows the largest positive momentum during the ranking period
    # if volume_filter is applied, filter out stocks whose trading volume on the previous day is below
    # its average in the ranking period
    def winners(self, date, ranking_period, n, volume_filter=False):
        """
        :param date: (str)
        :param ranking_period: length of ranking period (int)
        :param n: number of winners to pick (int)
        :param volume_filter: whether to apply volume filter on top of momentum indicator (bool)
        :return a list of tickers str[]
        """
        winners = []
        
        end = datetime.strptime(date, '%Y-%m-%d')
        
        #datetime for end of ranking_period
        start = end - timedelta(days=ranking_period)
        
        #change times back to str
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
        

        #rank stocks based on momentum
        for s in self.stocks:
            for i in range(n):
                if start_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    start -= timedelta(days=1)
                    start_date = start.strftime("%Y-%m-%d")
            if start_date not in self.stocks[s].df['daily_return']:
                return []
            self.stocks[s].momentum = self.momentum(s, start_date, end_date, ranking_period)
        ordered = sorted(self.stocks.items(), key=lambda kv: kv[1].momentum, reverse = False)
        #ordered = self.stocks.sort_values(by='Momentum', ascending=False)
        
        #pick top n
        for (ticker, stock) in ordered:
            if len(winners) < n and stock.momentum > 0:
                if volume_filter == True:
                    prev_start = start - timedelta(days= n*3)
                    prev_start_date = prev_start.strftime("%Y-%m-%d")
                    for i in range(n):
                        if prev_start_date in stock.df['Volume']:
                            break
                        else:
                            prev_start -= timedelta(days=1)
                            prev_start_date = prev_start.strftime("%Y-%m-%d")
                        
                    #prev_end = start
                    prev_end_date = start_date
  
                    if prev_start_date not in stock.df['Volume']:
                        return []
                    
                    prev_vol = np.mean(stock.df['Volume'].loc[prev_start_date:prev_end_date].dropna())
                    last_vol = np.mean(stock.df['Volume'].loc[start_date:end_date].dropna())

                    if last_vol > prev_vol:
                        winners += [ticker]
                else:
                    winners += [ticker] 
        return winners
                

    # //TODO:
    # pick n stock that shows the most negative momentum during the ranking period
    # if volume_filter is applied, filter out stocks whose trading volume on the previous day is above
    # its average in the ranking period
    def losers(self, date, ranking_period, n, volume_filter=False):
        """
        :param date: (str)
        :param ranking_period: length of ranking period (int)
        :param n: number of winners to pick (int)
        :param volume_filter: whether to apply volume filter on top of momentum indicator (bool)
        :return a list of tickers str[]
        """
        losers = []
        
        end = datetime.strptime(date, '%Y-%m-%d')
        
        #datetime for end of ranking_period
        start = end - timedelta(days=ranking_period)
        
        #change times back to str
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
        

        #rank stocks based on momentum
        for s in self.stocks:
            for i in range(n):
                if start_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    start -= timedelta(days=1)
                    start_date = start.strftime("%Y-%m-%d")
            if start_date not in self.stocks[s].df['daily_return']:
                return []
            self.stocks[s].momentum = self.momentum(s, start_date, end_date, ranking_period)
        ordered = sorted(self.stocks.items(), key=lambda kv: kv[1].momentum, reverse = True)
        #ordered = self.stocks.sort_values(by='Momentum', ascending=False)
        
        #pick top n
        for (ticker, stock) in ordered:
            if len(losers) < n and stock.momentum < 0:
                if volume_filter == True:
                    prev_start = start - timedelta(days= n*3)
                    prev_start_date = prev_start.strftime("%Y-%m-%d")
                    for i in range(n):
                        if prev_start_date in stock.df['Volume']:
                            break
                        else:
                            prev_start -= timedelta(days=1)
                            prev_start_date = prev_start.strftime("%Y-%m-%d")
                        
                    #prev_end = start
                    prev_end_date = start_date
  
                    if prev_start_date not in stock.df['Volume']:
                        return []
                    
                    prev_vol = np.mean(stock.df['Volume'].loc[prev_start_date:prev_end_date].dropna())
                    last_vol = np.mean(stock.df['Volume'].loc[start_date:end_date].dropna())

                    if last_vol > prev_vol:
                        losers += [ticker]
                else:
                    losers += [ticker] 
        return losers

    def get_stock(self, ticker):
        return self.stocks[ticker]

    def stock_price(self, ticker, date):
        if ticker == 'cash': return 1
        return self.stocks[ticker].df['Adj Close'][date]


    # //TODO: add/implement other methods related to the analyzation part


if __name__ == "__main__" : 
     tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'JNJ', 'JPM', 'BILI']
     b = Analyzer(tickers, "2019-01-01")
     
     c = b.winners("2019-10-01", 20, 5, True)
     d = b.winners("2019-10-01", 20, 5)
     print(c, d)
     
     e = b.winners("2019-06-01", 20, 5, True)
     f = b.winners("2019-06-01", 20, 5)
     print(e, f)
     
     g = b.losers("2019-06-01", 20, 5, True)
     h = b.losers("2019-06-01", 20, 5)
     print(g, h)
