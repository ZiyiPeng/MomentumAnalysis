import concurrent.futures
import os

from Stock import Stock
from scipy.stats import linregress
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import helper
from get_all_tickers import get_tickers as gt

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

    def momentum(self, s, start_idx, end_idx):
        """
        :param s: stock
        :param start_idx: index for start of ranking_period
        :param end_idx: index for end of ranking_period
        :return the momentum based on returns
        """
        returns = self.stocks[s].df['daily_return'].iloc[start_idx:end_idx].dropna()
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        momentum = ((1 + slope) ** 252) * (rvalue ** 2) # annualize slope and multiply by R^2
        #if s == "TSLA":
            #print(start_idx, end_idx, returns)
        return (returns, momentum)
    
    # //TODO:
    # pick n stock that shows the largest positive momentum during the ranking period
    # if volume_filter is applied, filter out stocks whose trading volume on the previous day is below
    # its average in the ranking period
    def winners(self, date, ranking_period, n, volume_filter=False):
        """
        :param date: (str) (Assumes that this is a valid date in our dataframe)
        :param ranking_period: length of ranking period (int)
        :param n: number of winners to pick (int)
        :param volume_filter: whether to apply volume filter on top of momentum indicator (bool)
        :return a list of tickers str[]
        """
        winners = []
        
        #rank stocks based on momentum
        for s in self.stocks:
            try:
                ranking_end = self.stocks[s].df.index.get_loc(date)
                ranking_start = ranking_end - ranking_period
                if ranking_end < ranking_period :
                    return []
                (returns, momentum) = self.momentum(s, ranking_start, ranking_end)
                self.stocks[s].momentum = momentum
                self.stocks[s].returns = np.mean(returns)
            except KeyError:
                self.stocks[s].momentum = None
        good_stocks = [s for s in self.stocks if self.stocks[s].momentum != None]
        ordered = sorted(good_stocks, key=lambda s: self.stocks[s].momentum, reverse = False)
        
        #pick top n
        for ticker in ordered:
            if len(winners) < n:
                if volume_filter == True:
                    
                    prev_start = ranking_start - (ranking_period*3)
                    prev_end = ranking_start
                    
                    if prev_end < (ranking_period *3) :
                        return []

                    prev_vol = np.mean(self.stocks[ticker].df['Volume'].iloc[prev_start:prev_end].dropna())
                    last_vol = np.mean(self.stocks[ticker].df['Volume'].iloc[ranking_start:ranking_end].dropna())

                    if last_vol > prev_vol:
                        winners += [ticker]
                else:
                    winners += [ticker] 

        return winners

    # pick n stock that shows the largest positive momentum during the ranking period
    # if volume_filter is applied, filter out stocks whose trading volume on the previous day is below
    # its average in the ranking period
    def losers(self, date, ranking_period, n, volume_filter=False):
        """
        :param date: (str) (Assumes that this is a valid date in our dataframe)
        :param ranking_period: length of ranking period (int)
        :param n: number of winners to pick (int)
        :param volume_filter: whether to apply volume filter on top of momentum indicator (bool)
        :return a list of tickers str[]
        """
        losers = []
        
        for s in self.stocks:
            try:
                ranking_end = self.stocks[s].df.index.get_loc(date)
                ranking_start = ranking_end - ranking_period
                if ranking_end < ranking_period :
                    return []
                (returns, momentum) = self.momentum(s, ranking_start, ranking_end)
                self.stocks[s].momentum = momentum
                self.stocks[s].returns = np.mean(returns)
            except KeyError:
                self.stocks[s].momentum = None
        good_stocks = [s for s in self.stocks if self.stocks[s].momentum != None]
        ordered_R = sorted(good_stocks, key=lambda s: self.stocks[s].momentum, reverse = True)
        
        #pick top n
                    
        for ticker in ordered_R:
            if len(losers) < n:
                if volume_filter == True:
                    prev_start = ranking_start - (ranking_period*3)
                    prev_end = ranking_start

                    if prev_end < (ranking_period *3) :
                        return []
                    
                    prev_vol = np.mean(self.stocks[ticker].df['Volume'].iloc[prev_start:prev_end].dropna())
                    last_vol = np.mean(self.stocks[ticker].df['Volume'].iloc[ranking_start:ranking_end].dropna())

                    if last_vol > prev_vol:
                        losers += [ticker]
                else:
                    losers += [ticker] 
        return losers

    def get_stock(self, ticker):
        return self.stocks[ticker]

    def stock_price(self, ticker, date):
        return self.stocks[ticker].df['Adj Close'][date]


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
    
    def calc_returns_momentums_volumes(self, date, ranking_period, test_stocks):
            
        for s in test_stocks:
            ranking_end = self.stocks[s].df.index.get_loc(date)
            ranking_start = ranking_end - ranking_period
            hold_start = ranking_end
            hold_end = hold_start + ranking_period
            break
        
        assert ranking_end >= ranking_period, "Unable to find ranking period"
        
        try:
            self.stocks[s].df['daily_return'].iloc[hold_start:hold_end]
        except AssertionError:
            print("Unable to find holding period")
        
        test_returns = []
        test_momentums = []
        test_volumes = []
        for s in test_stocks:
            test_returns += [np.mean(self.stocks[s].df['daily_return'].iloc[hold_start:hold_end].dropna())]
            (_, momentums) = self.momentum(s, ranking_start, ranking_end)
            test_momentums += [momentums]
            test_volumes += [np.mean(self.stocks[s].df['Volume'].iloc[ranking_start:ranking_end].dropna())]
        return test_returns, test_momentums, test_volumes
    
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
        (test_returns, _, _) = self.calc_returns_momentums_volumes(date, ranking_period, test_stocks)
            
        spy_prices = helper.get_historical_data("SPY", date, None)['Adj Close']
        all_spy_returns = np.log(spy_prices.pct_change() + 1)
        
        spy_return = np.mean(all_spy_returns[:ranking_period].dropna())
    
        (_, p_value) = stats.ttest_1samp(a=test_returns, popmean=spy_return)

        return p_value
    
    def plot_holding(self, date, test_hold, test_stocks):
        p_values = []
        hold = []
        for i in range(60, test_hold):
            try:
                p_value = self.t_test(date, i, test_stocks)
            except AssertionError:
                pass 
            hold += [i]
            p_values += [p_value]
        plt.plot(hold, p_values)
        plt.hlines(0.05, 0, test_hold, color = 'red', label = 'p-value = 0.05')
        plt.legend()
        plt.title("Number of Holding days vs p_values")# for the selected group
        plt.xlabel("Holding days")
        plt.ylabel("p_value")
        plt.show()
        
    def plot_momentum(self, test_stocks, date, ranking_period):
        returns = []
        momentums = []
        volumes = []
        length_ranking_period = []
        for i in range(1, ranking_period):
            try:
                (test_returns, test_momentums, test_volumes) = self.calc_returns_momentums_volumes(date, i, test_stocks)
                momentums += test_momentums
                returns += test_returns
                volumes += test_volumes
                length_ranking_period += [i]
            except AssertionError:
                pass 
        #fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(length_ranking_period, momentums, returns)
        ax.set_title("Momentums vs Returns for different ranking_periods")
        plt.show()
        
        ax2 = plt.axes(projection='3d')
        ax2.scatter3D(length_ranking_period, volumes, returns)
        ax2.set_title("Volumes vs Returns for different ranking_periods")
        plt.show()
        '''
        plt.plot(momentums, returns, 'o')
        plt.title("Momentums vs Returns for different ranking_periods")
        plt.xlabel("Momentum")
        plt.ylabel("Return")
        plt.show()
        plt.plot(volumes, returns,'o')
        plt.title("Volumes vs Returns for different ranking_periods")
        plt.xlabel("Volume")
        plt.ylabel("Return")
        plt.show()
        '''

if __name__ == "__main__" : 
     #tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'JNJ', 'JPM', 'BILI', 'TSLA']
     tickers = gt.get_biggest_n_tickers(40)
     b = Analyzer(tickers, "2010-01-01")
     
     w1 = b.winners("2010-02-09", 25, 5)
     l1 = b.losers("2010-02-09", 25, 5)
     
     #b = Analyzer(tickers, "2012-01-01")
     #w2 = b.winners("2013-08-01", 20, 5, True)
    # l2 = b.losers("2013-08-01", 20, 5, True)
     
     
     print(w1, l1)
     #print(w2, l2)


     #length of holding period = ranking period
     #test seleted stock momentums
     #print(b.t_test_momentum(tickers, d))
     
     #test holding period returns
     #print(b.t_test("2020-09-01", 20, w1))


     #b.plot_holding("2020-09-01", 70, w1)
     
    # b.plot_momentum(['AAPL'], "2020-09-01", 255)

