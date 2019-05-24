import pandas as pd
import numpy as np

def baseline1(unit,money,cross_array, price_array,limit_sell,stop_loss):
    principal = money
    money_over_time = []
    cond = 1
    annual_profit = 0.0
    for i in range(len(price_array)):
        #Entering position
        if cross_array[i] == 1 and cond == 1:
            units =  money/price_array[i]
            #Limit Sell
            upper = price_array[i]*(1+limit_sell)
            #Stop Loss
            lower = price_array[i]*(1-stop_loss)
            cond = 2
        elif cross_array[i] == -1 and cond == 1:
            units =  -money/price_array[i]
            #Limit Sell
            upper = price_array[i]*(1+stop_loss)
            #Stop Loss
            lower = price_array[i]*(1-limit_sell)
            cond = 3
        #Exit Condition
        if cond == 2:
            if cross_array[i] == -1 or price_array[i] >= upper or price_array[i] <= lower:
                money = (units) * price_array[i]
                cond = 1

        if cond == 3:
            if cross_array[i] == 1 or price_array[i] >= upper or price_array[i] <= lower:
                money = money*2 + (units) * price_array[i]
                cond = 1

        money_over_time.append(money)

    total_profit = money - principal
    #Assuming 252 trading days in a year
    annual_profit = ((money/principal) ** (252.0/len(price_array))-1)*100

    return money, money_over_time, total_profit, annual_profit

def baseline2(unit,money,cross_array, price_array):
    units = 0.0
    money = 10000.0
    principal = money
    money_over_time = []
    cond = 1
    annual_profit = 0.0
    for i in range(len(price_array)):
        #Entering position
        if cross_array[i] == 1 and cond == 1:
            units =  money/price_array[i]
            cond = 2
        elif cross_array[i] == -1 and cond == 1:
            units = -money/price_array[i]
            cond = 3
        #Exit Condition
        if cond == 2:
            if cross_array[i] == -1:
                money = (units) * price_array[i]
                cond = 1

        if cond == 3:
            if cross_array[i] == 1:
                money = money*2 + (units) * price_array[i]
                cond = 1

        money_over_time.append(money)

    total_profit = money - principal
    #Assuming 252 trading days in a year
    annual_profit = ((money/principal) ** (252.0/len(price_array))-1)*100

    return money, money_over_time, total_profit, annual_profit

def baseline3(unit,money,cross_array, price_array,limit_sell,stop_loss):
    principal = money
    money_over_time = []
    cond = 1
    annual_profit = 0.0
    for i in range(len(price_array)):
        #Entering Position
        if cross_array[i] == 1 and cond == 1:
            units =  money/price_array[i]
            cond = 2
            lower = price_array[i]*(1-stop_loss)
            if i != 0 and price_array[i] > price_array[i-1]:
                lower = price_array[i]*(1-stop_loss)
        elif cross_array[i] == -1 and cond == 1:
            units = -money/price_array[i]
            cond = 3
            upper = price_array[i]*(1+stop_loss)
            if i != 0 and price_array[i] < price_array[i-1]:
                upper = price_array[i]*(1+stop_loss)
        #Exiting Position
        if cond == 2:
            if cross_array[i] == -1 or price_array[i] <= lower:
                money = (units) * price_array[i]
                cond = 1
        if cond == 3:
            if cross_array[i] == 1 or price_array[i] >= upper:
                money = money*2 + units * price_array[i]
                cond = 1
        money_over_time.append(money)

    total_profit = money - principal
    #Assuming 252 trading days in a year
    annual_profit = ((money/principal) ** (252.0/len(price_array))-1)*100

    return money, money_over_time, total_profit, annual_profit

def sharpe_calc(money_over_time_array):

    return_array = []
    for i in range(len(money_over_time_array)):
        if i != len(money_over_time_array) - 1:
            return_array.append() = money_over_time_array[i+1]/money_over_time_array[i]-1

    sharpe = sqrt(252)*np.mean(return_array)/np.std(return_array)

    return sharpe

def main():

features_directory = 'data'
    for i, filename in enumerate(os.listdir(features_directory)):
        if i == 1 and not re.match(filename, ".DS_Store"):
            list_of_df.append(pd.read_csv('./data/Gold_6.csv'))

    to_use = ["SMA5Cross", "SMA10Cross","SMA15Cross", "SMA20Cross", "SMA50Cross", "SMA100Cross",
            "EMA10Cross", "EMA12Cross", "EMA20Cross", "EMA26Cross", "EMA50Cross", "EMA100Cross"]

    take_profit = 0.1
    cut_loss = 0.05

    for cross in to_use:
        print('='*25)
        print('Cross:', cross)
        for df in list_of_df:
            #Baseline1: Fixing Stop Losses and Limt Sell
            print('Baseline Strategy 1: Trading based on crosses with fixed stop losses and limit sells')
            money, money_array, total_profit, annual_profit = baseline1(unit = 0.0,money = 100, cross_array = df[cross], price_array = df["Settle_Price"],limit_sell = take_profit,stop_loss = cut_loss)

            #Baseline2: Sell only by crosses
            print('Baseline Strategy 2: Trading only based on crosses')
            money, money_array, total_profit, annual_profit = baseline2(unit = 0.0,money = 100,cross_array = df[cross], price_array = df["Settle_Price"])

            #Baseline3: Moving Stop losses and limit sell
            print('Baseline Strategy 3: Trading based on crosses with moving stop losses and limit sells')
            money, money_array, total_profit, annual_profit = baseline3(unit = 0.0,money = 100,cross_array = df[cross], price_array = df["Settle_Price"],limit_sell = take_profit,stop_loss = cut_loss)

        print('Baseline Strategy1 Sharpe Ratio:',)
        print('Baseline Strategy2 Sharpe Ratio:',)
        print('Baseline Strategy3 Sharpe Ratio:',)
        print('='*25)

if __name__ == "__main__":
    main()
