import pandas as pd
import numpy as np

df = pd.read_csv('./data/Gold_6.csv')

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
            if cross_array[i] == -1 or price_array[i] >= 1.07 * lastprice or price_array[i] <= 0.95 * lastprice:
                money = (units) * price_array[i]
                units =  0
                cond = 1
        money_over_time.append(money)
        units_over_time.append(units)
        i = i + 1;
    return money, money_over_time, units_over_time

def main():
    to_use = ["SMA5Cross", "SMA10Cross","SMA15Cross", "SMA20Cross", "SMA50Cross", "SMA100Cross",
            "EMA10Cross", "EMA12Cross", "EMA20Cross", "EMA26Cross", "EMA50Cross", "EMA100Cross"]
    for cross in to_use:
        money, money_array, unit_arry = buyCrossStrat(df[cross],df["Settle_Price"])
        print(str(cross) + "Total Profit", money - 10000)

if __name__ == "__main__":
    main()
