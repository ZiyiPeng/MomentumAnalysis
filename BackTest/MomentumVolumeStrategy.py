from datetime import datetime

from BackTest.TradingRecord import TradingRecord
from BackTest.TradingStrategy import TradingStrategy

class MomentumVolumeStrategy(TradingStrategy):
    def __init__(self, analyzer, init_value, period):
        super().__init__(analyzer, init_value, period)

    # override apply method in parent class, fill out trading records
    # self.records = {ticker: Records[]}
    def apply(self, start_date):
        period, holding_period = self.period, self.period
        end_date = self.df.index[-1]
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        capacity = self.initial_value
        while current_date < end_date:
            winners = self.analyzer.winners(current_date.strftime("%Y-%m-%d"), period, 5, True)
            losers = self.analyzer.losers(current_date.strftime("%Y-%m-%d"), period, 5, True)
            capacity_delta = 0
            for s in winners:
                record_cap_delta = self.trading_record(s, current_date, 2 * capacity / float(len(winners)), True)
                self.records[s].extend(record_cap_delta['records'])
                capacity_delta += record_cap_delta['capacity_delta']
            for s in losers:
                record_cap_delta = self.trading_record(s, current_date, capacity / float(len(losers)), False)
                self.records[s].extend(record_cap_delta['records'])
                capacity_delta += record_cap_delta['capacity_delta']
            # recalculate purchasing power
            capacity += capacity_delta
            idx = self.df.index.get_loc(current_date) + self.period
            current_date = self.df.index[idx]
