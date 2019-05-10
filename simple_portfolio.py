import numpy as np
import cvxpy as cvx

class SimplePortfolio():
    def __init__(self, num_assets):
        self.n = num_assets
        pass

    def optimize(self, returns, desired_variance):
        cov = np.outer(returns, returns)
        w = cvx.Variable(self.n)
        objective = w.T * returns
        constraints = [
            w >= 0.0,
            cvx.sum(w)<=1.0,
            cvx.quad_form(w,cov) <= desired_variance
        ]
        print(returns)
        print(cov)
        prob = cvx.Problem(cvx.Maximize(objective), constraints)
        prob.solve()
        return w.value
