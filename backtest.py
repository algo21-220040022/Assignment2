# -*- coding: utf-8 -*-

# 构建组合
import os
from parameters import FACTOR_CATEGORY
import pandas as pd 
import tushare as ts
import numpy as np
from empyrical import max_drawdown
import matplotlib.pyplot as plt
ts.set_token('e4fac0005c24fd30161f8a9455e1f0cdecf7ea0c75c66da27e15cb34')
pro = ts.pro_api()

class BackTest():
    def __init__(self, start_date="20130628", end_date="20210226", frequency="m", benchmark="000906.SH"):
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.stock_num = 15  #best:15
        self.benchmark_data = pro.index_monthly(ts_code=benchmark, start_date=start_date, end_date=end_date, fields='trade_date,close')
        self.benchmark_data.rename(columns={"trade_date":"date"}, inplace=True)
        self.benchmark_data["date"] = pd.to_datetime(self.benchmark_data["date"])
        self.benchmark_data.sort_values(by="date", ascending=True, inplace=True)
        self.benchmark_data["return"] = self.benchmark_data["close"].pct_change()
        self.holding_result = pd.DataFrame()

    def load_stock_data(self):
        self.all_stock_data = pd.read_pickle("./nero_factors_data_old_1/all_factor_data.pkl")
        # all_csv = os.listdir("./nero_factors_data/")
        # self.all_stock_data = pd.DataFrame()
        # for csv in all_csv:
        #     yearly_factor_data = pd.read_csv(f"./nero_factors_data/{csv}", index_col=0)
        #     self.all_stock_data = pd.concat([self.all_stock_data, yearly_factor_data], axis=0)
        # self.all_stock_data["date"] = pd.to_datetime(self.all_stock_data["date"])
        # stock_market_data = pd.read_csv("./data/raw_market_data.csv", index_col=0)
        # market_cap = pd.read_csv("./data/raw_daily_indicator_data.csv", index_col=0)
        # stock_market_data = stock_market_data.loc[:, ["ts_code", "trade_date", "close"]]
        # market_cap = market_cap.loc[:, ["ts_code", "trade_date", "total_mv"]]
        # stock_market_data = pd.merge(stock_market_data, market_cap, how="inner", on=["ts_code", "trade_date"])
        # stock_market_data.rename(columns={"ts_code": "code", "trade_date": "date"}, inplace=True)
        # stock_market_data["date"] = [pd.to_datetime(str(dt)) for dt in stock_market_data["date"].tolist()]
        # self.all_stock_data = pd.merge(self.all_stock_data, stock_market_data, on=["code", "date"])

    def load_all_trade_dates(self):
        all_trade_dates = pd.read_csv("./data/monthly_trading_dates.csv")
        self.all_trade_dates = [pd.to_datetime(dt) for dt in all_trade_dates["date"].tolist()]
        self.benchmark_ret = np.zeros(len(self.all_trade_dates))
        self.portfolio_ret = np.zeros(len(self.all_trade_dates))
        self.benchmark_ret[1:] = np.array(self.benchmark_data.loc[self.benchmark_data["date"].isin(self.all_trade_dates[1:])]["return"])

    def update_performance(self):
        num_month = len(self.all_trade_dates)-1
        self.portfolio_net_value = (1+self.portfolio_ret).cumprod()
        self.benchmark_net_value = (1+self.benchmark_ret).cumprod()
        self.maxDD = max_drawdown(self.portfolio_ret)
        self.excess_ret = self.portfolio_ret - self.benchmark_ret
        self.annual_excess_ret = (self.portfolio_net_value[-1]/self.benchmark_net_value[-1])**(12/num_month)-1
        self.cum_excess_ret = self.excess_ret.cumsum()
        self.volatility = np.std(self.portfolio_ret[1:])*np.sqrt(12)
        self.annual_ret = (self.portfolio_net_value[-1])**(12/num_month)-1
        self.information_ratio =self.annual_excess_ret / self.volatility
        self.holding_result.to_csv("./backtest_result/holding.csv")
        print(f"Annual Return:{self.annual_ret}; \nInformation Ratio:{self.information_ratio};\nmaxDD:{self.maxDD};"
              f"\nvolatility:{self.volatility}"
              f"\nSharpe:{(self.annual_ret - 0.03)/self.volatility}")

    def plot_result(self):
        result_pd = pd.DataFrame({"portfolio_net_value":self.portfolio_net_value, "benchmark":self.benchmark_net_value,
                                  "cum_excess_ret":self.cum_excess_ret}, index=self.all_trade_dates)
        result_pd.to_csv(f"./backtest_result/result.csv")

        result_pd.plot()
        plt.show()

    def run(self):
        self.load_stock_data()
        self.load_all_trade_dates()
        # factor_names = list(FACTOR_CATEGORY.keys()) #["value", "operation", "growth", "volume", "volatility"]
        factor_names = ["value", "operation", "growth", "volume", "momentum","volatility"]
        # factor_names = ["composite_factor"]
        for i,date,next_date in zip(range(len(self.all_trade_dates[:-1])),self.all_trade_dates[:-1], self.all_trade_dates[1:]):
            print(f"{date}")
            """Calculate the weight of each stock"""
            df_date = self.all_stock_data.loc[self.all_stock_data['date'] == date]
            df_date.loc[:, factor_names] = df_date.loc[:,factor_names].apply(lambda x: (x - x.mean())/x.std())

            df_date['Zi'] = df_date.loc[:, factor_names].sum(axis=1)
            df_date['Qi'] = df_date['Zi'].apply(lambda x: 1 + x if x > 0 else (1 - x) ** (-1))
            df_date['Qi*cap'] = df_date['Qi'] * df_date['total_mv']
            df_date = df_date.sort_values(by='Qi', ascending=False)
            df_date.reset_index(drop=True, inplace=True)

            df_date = df_date.iloc[:self.stock_num]
            df_date["weight"] = df_date["Qi*cap"] / df_date["Qi*cap"].sum()
            self.holding_result = pd.concat([self.holding_result,df_date.loc[:,["date","code","weight"]]], axis=0)

            """Update the return"""
            df_next_date = self.all_stock_data.loc[self.all_stock_data['date'] == next_date].loc[:,["code","close"]]
            df_next_date.rename(columns={"close":"next_close"}, inplace=True)
            df_date = pd.merge(df_date, df_next_date, on=["code"])
            print(df_date.shape)
            self.portfolio_ret[i+1] = (df_date["weight"]*(df_date["next_close"]-df_date["close"])/df_date["close"]).sum()
        print(factor_names)
        self.update_performance()
        self.plot_result()


if __name__ == "__main__":
    my_backtest = BackTest()
    my_backtest.run()






    







