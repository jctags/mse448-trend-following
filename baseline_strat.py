import pandas as pd
import numpy as np

df = pd.read_csv('./data/Gold_6.csv')
df = df[["EMA100Cross","Settle_Price"]]
df.head()

def buyCrossStrat(cross_array, price_array):
    units = 0
    money = 10000
    money_over_time = []
    units_over_time = []
    i = 0;
    cond = 1;
    for row in df.iterrows():
        if cross_array[i] == 1 and cond == 1:
            units =  money/price_array[i]
            lastprice = price_array[i]
            cond = 2
        if cond == 2:
            if cross_array[i] == -1 or price_array[i] >= 1.10 * lastprice or price_array[i] <= 0.95 * lastprice:
                money = (units) * price_array[i]
                units =  0
                cond = 1
        money_over_time.append(money)
        units_over_time.append(units)
        i = i + 1;
    return money, money_over_time, units_over_time

def main():
    money, money_array, unit_arry = buyCrossStrat(df["EMA100Cross"],df["Settle_Price"])
    print(money)

if __name__ == "__main__":
    main()
