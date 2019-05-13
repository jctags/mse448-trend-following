import pandas as pd
import numpy as np

df = pd.read_csv('./Gold.csv')
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
        if df["EMA100Cross"][i] == 1 and cond == 1:
            units =  money/df["Settle_Price"][i]
            lastprice = df["Settle_Price"][i]
            cond = 2
        if cond == 2:
            if df["EMA100Cross"][i] == -1 or df["Settle_Price"][i] >= 1.10 * lastprice or df["Settle_Price"][i] <= 0.95 * lastprice:
                money = (units) * df["Settle_Price"][i]
                units =  0
                cond = 1
        money_over_time.append(money)
        units_over_time.append(units)
    #    lastprice_over_time.append(lastprice)
        i = i + 1;

    return money

if __name__ == "__main__":
    main()
