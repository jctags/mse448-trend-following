import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


#Import files
list_of_df = []
directory = 'data'
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        df = pd.read_csv('data/'+str(filename))
        list_of_df.append(df.tail(500))

for i in range(len(list_of_df)):
    plt.plot(list_of_df[i]['Daily_Return'])
    plt.show()
