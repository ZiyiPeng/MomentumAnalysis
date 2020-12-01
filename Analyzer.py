import concurrent.futures
import os

from Stock import Stock
from scipy.stats import linregress
import scipy.stats as stats
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

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
        #print(momentum)
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
            for i in range(ranking_period):
                if start_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    start -= timedelta(days=1)
                    start_date = start.strftime("%Y-%m-%d")
            for i in range(ranking_period):
                if end_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    end -= timedelta(days=1)
                    end_date = end.strftime("%Y-%m-%d")
            if start_date not in self.stocks[s].df['daily_return']:
                return []
            if end_date not in self.stocks[s].df['daily_return']:
                return []
            self.stocks[s].momentum = self.momentum(s, start_date, end_date, ranking_period)
        ordered = sorted(self.stocks.items(), key=lambda kv: kv[1].momentum, reverse = False)
        #ordered = self.stocks.sort_values(by='Momentum', ascending=False)
        
        #pick top n
        for (ticker, stock) in ordered:
            if len(winners) < n:
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
            for i in range(ranking_period):
                if start_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    start -= timedelta(days=1)
                    start_date = start.strftime("%Y-%m-%d")
            for i in range(ranking_period):
                if end_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    end -= timedelta(days=1)
                    end_date = end.strftime("%Y-%m-%d")
            if start_date not in self.stocks[s].df['daily_return']:
                return []
            if end_date not in self.stocks[s].df['daily_return']:
                return []
            self.stocks[s].momentum = self.momentum(s, start_date, end_date, ranking_period)
        ordered = sorted(self.stocks.items(), key=lambda kv: kv[1].momentum, reverse = True)
        #ordered = self.stocks.sort_values(by='Momentum', ascending=False)
        
        #pick top n
        for (ticker, stock) in ordered:
            if len(losers) < n:
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
        return self.stocks[ticker][date]


    # //TODO: add/implement other methods related to the analyzation part
    
    '''
    #test if selected stocks have higher/lower returns than the population
    #look ahead at the holding period returns
    #assume that momentums have already been calculted
    #Null: spy mean and the test_stocks mean are the same
    #Alt: the means are different (p value < 0.05)
    def t_test_momentum(self, tickers, test_stocks):
        """
        :param date: (str)
        :param tickers: a list of tickers (str[])
        :param test_stocks: list of stocks that we want to test if significant
        :return p-value (float)
        """
        total_momentum = []
        test_momentum = []
        for s in tickers:
            total_momentum += [self.stocks[s].momentum]
            if s in test_stocks:
                test_momentum += [self.stocks[s].momentum]
            
        pop_mean_momentum = self.stocks['SPY'].momentum
    
        (_, p_value) = stats.ttest_1samp(a=test_momentum, popmean=pop_mean_momentum)

        return p_value
    '''
    
    
    #test if selected stocks have higher/lower returns than SPY
    #look ahead at the holding period returns
    #assume that momentums have already been calculted
    #Null: spy mean and the test_stocks mean are the same
    #Alt: the means are different (p value < 0.05)
    def t_test(self, date, ranking_period, test_stocks):
        """
        :param date: (str)
        :param ranking_period: length of ranking period (int)
        :param test_stocks: list of stocks that we want to test if significant
        :return p-value (float)
        """
        start = datetime.strptime(date, '%Y-%m-%d')
        
        #datetime for end of ranking_period
        end = start + timedelta(days=ranking_period)
        
        #change times back to str
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")
        
        test_returns = []
        for s in test_stocks:
            for i in range(ranking_period):
                if start_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    start -= timedelta(days=1)
                    start_date = start.strftime("%Y-%m-%d")
            for i in range(ranking_period):
                if end_date in self.stocks[s].df['daily_return']:
                    break
                else:
                    end -= timedelta(days=1)
                    end_date = end.strftime("%Y-%m-%d")
            assert start_date in self.stocks[s].df['daily_return'],"Invalid start date"
            assert end_date in self.stocks[s].df['daily_return'],"Invalid end date"
            test_returns += [np.mean(self.stocks[s].df['daily_return'].loc[start_date:end_date].dropna())]
            
        spy_return = np.mean(self.stocks[s].df['daily_return'].loc[start_date:end_date].dropna())
    
        (_, p_value) = stats.ttest_1samp(a=test_returns, popmean=spy_return)

        return p_value
    
    def plot_holding(self, date, test_hold, test_stocks):
        p_values = []
        hold = []
        for i in range(1, test_hold):
            try:
                p_value = self.t_test(date, i, test_stocks)
                hold += [i]
                p_values += [p_value]
            except AssertionError:
                pass 
        plt.plot(hold, p_values)
        plt.hlines(0.05, 0, test_hold, color = 'red', label = 'p-value = 0.05')
        plt.legend()
        plt.title("Number of Holding days vs p_values")
        plt.xlabel("Holding days")
        plt.ylabel("p_value")
        plt.show()
        

if __name__ == "__main__" : 
     tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'JNJ', 'JPM', 'BILI', 'SPY']
     b = Analyzer(tickers, "2019-01-01")
     
     c = b.winners("2019-02-01", 20, 5, True)
     d = b.winners("2019-02-01", 20, 5)
     print(c, d)
     
     e = b.winners("2019-06-01", 20, 5, True)
     f = b.winners("2019-06-01", 20, 5)
     print(e, f)
     
     g = b.losers("2019-06-01", 20, 5, True)
     h = b.losers("2019-06-01", 20, 5)
     print(g, h)
     
     #length of holding period = ranking period
     #test seleted stock momentums
     #print(b.t_test_momentum(tickers, d))
     
     #test holding period returns
     print(b.t_test("2019-06-01", 20, d))


     b.plot_holding("2019-06-01", 120, d)

