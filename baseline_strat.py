import pandas as pd
import numpy as np

df = pd.read_csv('./data/Gold_6.csv')

def baseline1(cross_array, price_array,limit_sell,stop_loss):
    units = 0.0
    money = 10000.0
    principal = money
    money_over_time = []
    cond = 1
    annual_profit = 0.0
    for i in range(len(price_array)):
        #Buy Condition
        if cross_array[i] == 1 and cond == 1:
            units =  money/price_array[i]
            #Limit Sell
            upper = price_array[i]*(1+limit_sell)
            #Stop Loss
            lower = price_array[i]*(1-stop_loss)
            cond = 2
        #Sell Condition
        if cond == 2:
            if cross_array[i] == -1 or price_array[i] >= upper or price_array[i] <= lower:
                money = (units) * price_array[i]
                cond = 1
        money_over_time.append(money)

    total_profit = money - principal
    #Assuming 252 trading days in a year
    annual_profit = ((money/principal) ** (252.0/len(price_array))-1)*100

    return money, money_over_time, total_profit, annual_profit

def baseline2(cross_array, price_array):
    units = 0.0
    money = 10000.0
    principal = money
    money_over_time = []
    cond = 1
    annual_profit = 0.0
    for i in range(len(price_array)):
        #Buy Condition
        if cross_array[i] == 1 and cond == 1:
            units =  money/price_array[i]
            cond = 2
        #Sell Condition
        if cond == 2:
            if cross_array[i] == -1:
                money = (units) * price_array[i]
                cond = 1
        money_over_time.append(money)

    total_profit = money - principal
    #Assuming 252 trading days in a year
    annual_profit = ((money/principal) ** (252.0/len(price_array))-1)*100

    return money, money_over_time, total_profit, annual_profit

def baseline3(cross_array, price_array,limit_sell,stop_loss):
    units = 0.0
    money = 10000.0
    principal = money
    money_over_time = []
    cond = 1
    annual_profit = 0.0
    for i in range(len(price_array)):
        #Buy Condition
        if cross_array[i] == 1 and cond == 1:
            units =  money/price_array[i]
            cond = 2
            lower = price_array[i]*(1-stop_loss)
            if i != 0 and price_array[i] > price_array[i-1]:
                lower = price_array[i]*(1-stop_loss)
        #Sell Condition
        if cond == 2:
            if cross_array[i] == -1 or price_array[i] <= lower:
                money = (units) * price_array[i]
                cond = 1
        money_over_time.append(money)

    total_profit = money - principal
    #Assuming 252 trading days in a year
    annual_profit = ((money/principal) ** (252.0/len(price_array))-1)*100

    return money, money_over_time, total_profit, annual_profit

def main():

    to_use = ["SMA5Cross", "SMA10Cross","SMA15Cross", "SMA20Cross", "SMA50Cross", "SMA100Cross",
            "EMA10Cross", "EMA12Cross", "EMA20Cross", "EMA26Cross", "EMA50Cross", "EMA100Cross"]

    take_profit = 0.1
    cut_loss = 0.05

    #Baseline1: Fixing Stop Losses and Limt Sell
    print('Baseline Strategy 1: Trading based on crosses with fixed stop losses and limit sells')
    for cross in to_use:
        money, money_array, total_profit, annual_profit = baseline1(cross_array = df[cross], price_array = df["Settle_Price"],limit_sell = take_profit,stop_loss = cut_loss)
        print(str(cross), 'Total Profit', total_profit)
        print(str(cross), 'Annaul Profit', annual_profit,'%')

    #Baseline2: Sell only by crosses
    print('Baseline Strategy 2: Trading only based on crosses')
    for cross in to_use:
        money, money_array, total_profit, annual_profit = baseline2(cross_array = df[cross], price_array = df["Settle_Price"])
        print(str(cross), 'Total Profit', total_profit)
        print(str(cross), 'Annaul Profit', annual_profit,'%')

    #Baseline3: Moving Stop losses and limit sell
    print('Baseline Strategy 3: Trading based on crosses with moving stop losses and limit sells')
    for cross in to_use:
        money, money_array, total_profit, annual_profit = baseline3(cross_array = df[cross], price_array = df["Settle_Price"],limit_sell = take_profit,stop_loss = cut_loss)
        print(str(cross), 'Total Profit', total_profit)
        print(str(cross), 'Annaul Profit', annual_profit,'%')

if __name__ == "__main__":
    main()
