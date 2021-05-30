from backtest import Backtest
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from strategy import Strategy
import traceback
from itertools import product
from scipy import interpolate
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from numpy import unravel_index
from scipy import ndimage
from pandas_datareader import data


class Optimization:

    def __init__(self, df):
        self.backtest = Backtest(df)
        self.backtest.freq='1Min'

    def f(self, x):
        x = [int(round(x[i])) if self.type[i] == 'int' else x[i] for i in range(len(x))]
        self.backtest.run(self.strategy(x))
        sr = -self.backtest.sharpe_ratio_year
        return sr

    def grid_smoothing_method(self, strategy,n_split=10):
        # compute the sharpe ratio on the meshgrid of the space,
        # use gaussian kernal to give a smooth estimate of the sharpe matrix.
        # find optimizations based on smoothed estimates.
        self.strategy = strategy
        self.strategy()
        self.bound = self.backtest.strategy.bound
        self.type = self.backtest.strategy.type
        self.bound_splits = np.linspace(self.bound[:,0], self.bound[:, 1], n_split)

        str_exc = 'product('+','.join(['self.bound_splits[:,' + str(i) +']' for i in range(len(self.bound))]) + ')'
        mesh_sr=eval(str_exc)
        str_exc = 'product('+','.join(['range(n_split)' for i in range(len(self.bound))]) + ')'
        mesh_iter = eval(str_exc)
        self.sr_matrix = np.zeros(shape=[len(e) for e in self.bound_splits.T])

        list_idx = [grid_idx for grid_idx in mesh_iter]
        list_grid = [grid for grid in mesh_sr]

        def process(i):
            print(float(i) / n_split ** len(self.bound))
            self.sr_matrix[list_idx[i]] = -self.f(list_grid[i])
        [process(i) for i in range(len(list_idx))]

        smooth_kernel = ndimage.gaussian_filter(self.sr_matrix, sigma=1)
        idx_opt = unravel_index(smooth_kernel.argmax(), smooth_kernel.shape)
        x_opt = [self.bound_splits[idx_opt[i],i] for i in range(len(idx_opt))]
        sr_opt = smooth_kernel[idx_opt]

        return x_opt, sr_opt, smooth_kernel


if __name__ == '__main__':
    start_date = '2016-01-01'
    end_date = '2021-04-01'
    panel_data = data.DataReader('GS', 'yahoo', start_date, end_date)
    opt = Optimization(panel_data)
    x_opt, sr_opt, smooth_kernel = opt.grid_smoothing_method(opt.backtest.strategy.strategy1, n_split=5)
    print(x_opt, sr_opt)