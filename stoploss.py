import numpy as np

def get_stoploss_allocation(allocation, previous_allocations, previous_returns, max_loss = 0.2):
    new_w = np.zeros(len(allocation))
    for i in range(len(allocation)):
        if np.abs(allocation[i])<0.001:
            new_w[i] = allocation[i]
            continue
        steps = 0
        for j in range(len(previous_allocations)):
            prev_weight = previous_allocations[-(1+j)][i]
            if np.abs(prev_weight) > 0.001 and prev_weight * allocation[i] > 0.0:
                steps += 1
        value_since_open = 1.0
        for k in range(steps):
            lookback = steps-k
            value_since_open = value_since_open * (1.0 + previous_returns.values[-lookback, i])
        return_since_open = value_since_open - 1.0
        if return_since_open * np.sign(allocation[i]) > -max_loss:
            new_w[i] = allocation[i]
        #else zero since initialized as zeros
    return new_w
