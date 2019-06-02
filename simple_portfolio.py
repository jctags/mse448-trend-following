import numpy as np
import cvxpy as cvx

class SimplePortfolio():
    def __init__(self, num_assets):
        self.n = num_assets

    def optimize(self, holdings, returns, cov, desired_variance, transaction_costs = 0.0):

        trades = cvx.Variable(self.n)

        new_portfolio = holdings + trades

        objective = new_portfolio.T * returns - cvx.sum(cvx.abs(trades)) * transaction_costs
        constraints = [
            #w >= 0.0,
            cvx.sum(cvx.abs(new_portfolio))<=1.0,
            #cvx.sum(w)>=0.0,
            cvx.quad_form(new_portfolio,cov) <= desired_variance
        ]
        prob = cvx.Problem(cvx.Maximize(objective), constraints)
        prob.solve()
        return trades.value + holdings
