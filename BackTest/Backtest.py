from Analyzer import Analyzer
from get_all_tickers import get_tickers as gt

from BackTest.MomentumVolumeStrategy import MomentumVolumeStrategy
from BackTest.SimpleMomentumStrategy import SimpleMomentumStrategy
from Stock import Stock

if __name__ == '__main__':
    tickers = gt.get_biggest_n_tickers(40)
    analyzer = Analyzer(tickers, start='2010-01-01')
    initial_investment = 10000.0
    period = 10
    sm_strt = MomentumVolumeStrategy(analyzer, initial_investment, 25)
    # use SPY as benchmark
    SPY = Stock('SPY')
    SPY.populate_df('2010-01-01')
    SPY.df['annual_return'] = SPY.df['daily_return'].rolling(252).sum()

    sm_strt.apply('2010-01-04')
    sm_strt.backtrace()
    sm_strt.plot_values(SPY)
    sm_strt.plot_daily_return(SPY)
    sm_strt.plot_annual_return(SPY)
    sm_strt.t_test(SPY)
    
    print('a')
