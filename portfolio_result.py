import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


#Plot
result = pd.read_csv('portfolio_result/combined.csv')
plt.plot(result['Baseline'], label = "Baseline Strategy")
plt.plot(result['LSTM'], label = "LSTM")
plt.plot(result['Linear'], label = "Linear Regression")
plt.plot(result['Neural'], label = "Neural Network")
plt.title("Portfolio Value Overtime of Different Strategies ", fontsize = 15)
plt.axis([0, 450, 0.90, 1.40])
plt.legend()
plt.xlabel('Index: Trading Days')
plt.ylabel('Portfolio Value')
plt.show()

#Statistics
