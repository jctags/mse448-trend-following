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
        add = df.tail(500)
        list_of_df.append(add)

return_dict = {}

for i in range(len(list_of_df)):
    return_dict["Asset" + str(i+1)] = list_of_df[i]['Daily_Return'].tolist()
    #plt.plot(list_of_df[i]['Daily_Return'])
    #plt.xlabel('Index: Trading Days')
    #plt.ylabel('Daily Return')
    #plt.savefig("Plot" + str(i))

cor_df = pd.DataFrame(data=return_dict)

def plot_corr(df,size=10):

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr,interpolation='nearest', cmap = "seismic", vmin = -1, vmax = 1)
    plt.title("Correlation of Returns of 36 Different Assets", fontsize = 24)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 70)
    ax.xaxis.set_ticks_position('bottom')
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    plt.show()



# alpha = ['ABC', 'DEF', 'GHI', 'JKL']
#
# data = np.random.random((4,4))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(data, interpolation='nearest')
# fig.colorbar(cax)
#
# plt.show()


plot_corr(cor_df)


# plt.matshow(cor_df.corr())
# plt.colorbar
# plt.show()
