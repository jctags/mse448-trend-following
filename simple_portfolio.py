import numpy as np
import cvxpy as cvx

class SimplePortfolio():
    def __init__(self, num_assets):
        self.n = num_assets

    def optimize(self, returns, cov, desired_variance):

        w = cvx.Variable(self.n)
        objective = w.T * returns
        constraints = [
            #w >= 0.0,
            cvx.sum(cvx.abs(w))<=1.0,
            #cvx.sum(w)>=0.0,
            cvx.quad_form(w,cov) <= desired_variance
        ]
        prob = cvx.Problem(cvx.Maximize(objective), constraints)
        prob.solve()
        return w.value
