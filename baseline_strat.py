import pandas as pd
import numpy as np
import os

def baseline1(unit,money,cross_array, price_array,limit_sell,stop_loss,leverage):
    principal = money
    money = money*leverage
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

def baseline2(unit,money,cross_array, price_array,leverage):
    principal = money
    money = money*leverage
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

def baseline3(unit,money,cross_array, price_array,limit_sell,stop_loss,leverage):
    principal = money
    money = money*leverage
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
            return_array.append(money_over_time_array[i+1]/money_over_time_array[i]-1)

    sharpe = np.sqrt(252)*np.mean(return_array)/np.std(return_array)
    avg_annual_profit = np.sqrt(252)*np.mean(return_array)

    return sharpe,avg_annual_profit

def main():

    leverage = 5
    nday = 500
    directory = 'data'
    starting_money = 1000000.0 #1million
    num_file = 0
    to_use = ["SMA5Cross", "SMA10Cross","SMA15Cross", "SMA20Cross", "SMA50Cross", "SMA100Cross",
        "EMA10Cross", "EMA12Cross", "EMA20Cross", "EMA26Cross", "EMA50Cross", "EMA100Cross"]
    take_profit = 0.1
    cut_loss = 0.05
    list_of_df = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            num_file += 1
            df = pd.read_csv('data/'+str(filename))
            list_of_df.append(df.tail(nday))

    strat1 = np.zeros(nday)
    strat2 = np.zeros(nday)
    strat3 = np.zeros(nday)

    for cross in to_use:
        print('*'*100)
        print('Cross:', cross)
        for df in list_of_df:
            #Baseline1: Fixing Stop Losses and Limt Sell
            money1, money_array1, total_profit2, annual_profit1 = baseline1(unit = 0.0,money = starting_money/len(list_of_df),
                cross_array = df[cross].tolist(), price_array = df["Settle_Price"].tolist(),limit_sell = take_profit,stop_loss = cut_loss,
                leverage = leverage)

            #Baseline2: Sell only by crosses
            money2, money_array2, total_profit2, annual_profit2 = baseline2(unit = 0.0,money = starting_money/len(list_of_df),cross_array = df[cross].tolist(),
                price_array = df["Settle_Price"].tolist(), leverage = leverage)

            #Baseline3: Moving Stop losses and limit sell
            money3, money_array3, total_profit3, annual_profit3 = baseline3(unit = 0.0,money = starting_money/len(list_of_df),cross_array = df[cross].tolist(),
                price_array = df["Settle_Price"].tolist(),limit_sell = take_profit,stop_loss = cut_loss, leverage = leverage)

            strat1 += money_array1
            strat2 += money_array2
            strat3 += money_array3

        print('Baseline Strategy 1: Trading based on crosses with fixed stop losses and limit sells')
        sharpe1,avg_annual_profit1 = sharpe_calc(strat1)
        print('Sharpe Ratio:',sharpe1)
        print('Profit:',avg_annual_profit1*100,'%')
        print('='*25)

        print('Baseline Strategy 2: Trading only based on crosses')
        sharpe2,avg_annual_profit2 = sharpe_calc(strat2)
        print('Sharpe Ratio:',sharpe2)
        print('Profit:',avg_annual_profit2*100,'%')
        print('='*25)

        print('Baseline Strategy 3: Trading based on crosses with moving stop losses and limit sells')
        sharpe3,avg_annual_profit3 = sharpe_calc(strat3)
        print('Sharpe Ratio:',sharpe3)
        print('Profit:',avg_annual_profit3*100,'%')


if __name__ == "__main__":
    main()
