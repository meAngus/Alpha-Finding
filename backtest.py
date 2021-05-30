import numpy as np
from alpha import Alpha
import pandas as pd
from strategy import *

class Backtest:

    def __init__(self, df):
        self.alpha = Alpha(df)
        self.strategy = Strategy(df)
        self.cost = 0.001
        self.freq = '1Min'

    def run(self, strategy):
        self.cash = 1
        self.stock = 0

        cond_b, cond_s, price_b, price_s = strategy

        def step(t):
            if cond_b[t] and self.stock == 0 and price_b[t] > self.alpha.low[t + 1]:
                self.stock = self.cash / price_b[t] * (1 - self.cost)
                self.cash = 0

            elif cond_s[t] and self.cash == 0 and price_s[t] < self.alpha.high[t + 1]:
                self.cash = self.stock * price_s[t] * (1 - self.cost)
                self.stock = 0

            return self.cash + self.stock*self.alpha.close[t+1]
        #  backtest
        self.history_asset = [step(t) for t in range(1, self.alpha.T-1)]
        self.r = np.diff(self.history_asset)/self.history_asset[:-1]
        self.sharpe_ratio = np.average(self.r)/np.std(self.r)

        if self.freq == '1Min':
            self.sharpe_ratio_year = self.sharpe_ratio*np.sqrt(240*250)
        elif self.freq == '5Min':
            self.sharpe_ratio_year = self.sharpe_ratio*np.sqrt(240*250/5)
        elif self.freq == '10Min':
            self.sharpe_ratio_year = self.sharpe_ratio * np.sqrt(240 * 250 / 10)
        elif self.freq == '15Min':
            self.sharpe_ratio_year = self.sharpe_ratio * np.sqrt(240 * 250 / 15)
        elif self.freq == '30Min':
            self.sharpe_ratio_year = self.sharpe_ratio * np.sqrt(240 * 250 / 30)
        elif self.freq == '60Min':
            self.sharpe_ratio_year = self.sharpe_ratio * np.sqrt(240 * 250 / 60)


        if np.isnan(self.sharpe_ratio):
            self.sharpe_ratio = -1
            self.sharpe_ratio_year = -1


if __name__ == '__main__':
    df = pd.read_csv('./data/^DJI.csv')
    backtest = Backtest(df)
    backtest.run(backtest.strategy.strategy1())
    print(backtest.sharpe_ratio_year)







