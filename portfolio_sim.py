import numpy as np
import matplotlib.pyplot as plt

class PortfolioSimulator(object):

    def __init__(self, opt, n):
        self.portfolio_value = 1.0
        self.portfolio_over_time = [self.portfolio_value]
        self.portfolio_returns = []
        self.naive_value = 1.0
        self.naive_over_time = [self.naive_value]
        self.naive_returns = []
        self.naive_allocation = np.ones(n)/n
        self.allocations = [np.zeros(n)]
        self.opt = opt #optimizer

    def simulate(self, pred_df, daily_returns, cov, desired_variance, transaction_costs = 0.0):

        for i in range(len(pred_df)):
            naive_return = np.dot(self.naive_allocation, daily_returns.iloc[i,:].values)
            self.naive_value *= (1+naive_return)
            self.naive_over_time.append(self.naive_value)
            self.naive_returns.append(naive_return)

            w = self.opt.optimize(self.allocations[-1], pred_df.iloc[i, :].values, cov, desired_variance, transaction_costs)
            self.allocations.append(w)
            p_return = np.dot(w, daily_returns.iloc[i, :].values)

            trade_pct = np.sum(np.abs(self.allocations[-1] - self.allocations[-2]))
            p_return = p_return - trade_pct * transaction_costs

            self.portfolio_value *= (1+p_return)
            self.portfolio_over_time.append(self.portfolio_value)
            self.portfolio_returns.append(p_return)

    def plot_over_time(self):
        plt.plot(self.portfolio_over_time)
        plt.plot(self.naive_over_time)
        plt.legend(['Portfolio', 'Naive Strategy'])
        plt.show()

    def get_longer_frame_returns(self, steps, use_portfolio=True):
        output = []
        if use_portfolio:
            returns = self.portfolio_returns
        else:
            returns = self.naive_returns

        for i in range(len(returns)-steps+1):
            value = 1.0
            for j in range(steps):
                value = value * (1.0+returns[i+j]);
            value = value-1.0
            output.append(value)

        return output
