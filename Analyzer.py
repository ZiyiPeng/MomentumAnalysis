import concurrent.futures
import os

from Stock import Stock
from scipy.stats import linregress
from datetime import datetime, timedelta
import numpy as np

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
        winners = [None] * n
        
        start = datetime.strptime(date, '%Y-%m-%d')
        
        #datetime for end of ranking_period
        end = start + timedelta(days=ranking_period)
        
        #change times back to str
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
            
        #rank stocks based on momentum
        for s in self.stocks:
            while end_date not in self.stocks[s].df['daily_return']:
                end += timedelta(days=1)
                end_date = end.strftime("%Y-%m-%d")
            self.stocks[s].momentum = self.momentum(s, start_date, end_date, ranking_period)
        ordered = sorted(self.stocks.items(), key=lambda kv: kv[1].momentum, reverse = False)
        #ordered = self.stocks.sort_values(by='Momentum', ascending=False)
        
        #pick top n
        i = 0
        for (ticker, stock) in ordered:

            if i < n and stock.momentum > 0:
                winners[i] = ticker 
                if volume_filter == True:
                    prev = start - timedelta(days=1)
                    prev_date = prev.strftime("%Y-%m-%d")
                    while prev_date not in stock.df['Volume']:
                        prev -= timedelta(days=1)
                        prev_date = prev.strftime("%Y-%m-%d")
                    average_volume = np.mean(stock.df['Volume'].loc[start_date:end_date].dropna())
                    prev_volume = stock.df['Volume'][prev_date]
                    if prev_volume < average_volume:
                        winners[i] = None
            i+=1
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
        losers = [None] * n
        
        start = datetime.strptime(date, '%Y-%m-%d')
        
        #datetime for end of ranking_period
        end = start + timedelta(days=ranking_period)
        
        #change times back to str
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
            
        #rank stocks based on momentum
        for s in self.stocks:
            while end_date not in self.stocks[s].df['daily_return']:
                end += timedelta(days=1)
                end_date = end.strftime("%Y-%m-%d")
            self.stocks[s].momentum = self.momentum(s, start_date, end_date, ranking_period)
        ordered = sorted(self.stocks.items(), key=lambda kv: kv[1].momentum, reverse = True)
        #ordered = self.stocks.sort_values(by='Momentum', ascending=False)
        
        #pick top n
        i = 0
        for (ticker, stock) in ordered:

            if i < n and stock.momentum > 0:
                losers[i] = ticker 
                if volume_filter == True:
                    prev = start - timedelta(days=1)
                    prev_date = prev.strftime("%Y-%m-%d")
                    while prev_date not in stock.df['Volume']:
                        prev -= timedelta(days=1)
                        prev_date = prev.strftime("%Y-%m-%d")
                    average_volume = np.mean(stock.df['Volume'].loc[start_date:end_date].dropna())
                    prev_volume = stock.df['Volume'][prev_date]
                    if prev_volume > average_volume:
                        losers[i] = None
            i+=1
        return losers 

    def get_stock(self, ticker):
        return self.stocks[ticker]

    def stock_price(self, ticker, date):
        return self.stocks[ticker][date]


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
     
     g = b.losers("2019-06-01", 20, 5)
     h = b.losers("2019-06-01", 20, 5, True)
     print(g, h)