# import pandas as pd
"""
# merge data
raw_daily_indicator_data = pd.read_csv("./data/raw_daily_indicator_data.csv", index_col=0)
raw_daily_indicator_data.rename(columns={'trade_date':'date','ts_code':'code'}, inplace=True)

raw_market_data = pd.read_csv("./data/raw_market_data.csv", index_col=0)
raw_market_data.rename(columns={'trade_date':'date','ts_code':'code'}, inplace=True)

raw_daily_indicator_data['date'] = [str(date)[:8] for date in raw_daily_indicator_data['date'].tolist()]
raw_market_data['date'] = [str(date)[:8] for date in raw_market_data['date'].tolist()]

raw_daily_indicator_data.drop(columns=['close'], inplace=True)
raw_daily_data = pd.merge(left=raw_market_data, right=raw_daily_indicator_data, on=['code','date'])
raw_daily_data.to_pickle("./data/raw_daily_data.pkl")
"""
#import os

# merge factor data of all stocks
# raw_factor_data = pd.DataFrame()
# stock_factor_pkls = os.listdir("./raw_stock_factors/")
# j = 1
# for i,pkl in enumerate(stock_factor_pkls):
#     print(f"Load data of {pkl[:6]}....{i+1}/1496")
#     factor_data = pd.read_pickle(f"./raw_stock_factors/{pkl}")
#     raw_factor_data = pd.concat([raw_factor_data, factor_data], axis=0)
#     if (i+1) %300 == 0:
#         raw_factor_data.to_pickle(f"./raw_stock_factors/raw_factor_data_{j}.pkl")
#         raw_factor_data = pd.DataFrame()
#         j += 1
# raw_factor_data.to_pickle(f"./raw_stock_factors/raw_factor_data_{j}.pkl")
#import time

"""For train data"""
# import pandas as pd
# from dateutil.relativedelta import relativedelta
# import tushare as ts
#
# ts.set_token('e4fac0005c24fd30161f8a9455e1f0cdecf7ea0c75c66da27e15cb34')
# pro = ts.pro_api()
# def get_last_end_trade_date(start_date, end_date, period=20):
#     start_dt = str(start_date)[:10].replace("-", "")
#     end_dt = str(end_date)[:10].replace("-", "")
#     df = pro.query('trade_cal', start_date=start_dt, end_date=end_dt)
#     all_trade_dates = df.loc[df["is_open"]==1]
#     end_trade_date_str = all_trade_dates.iloc[-1]["cal_date"]
#     return pd.to_datetime(all_trade_dates.iloc[-period-2]["cal_date"]), end_trade_date_str
#     # return df["is_open"].value_counts()[1]
#
# train_period_list = []
# start_date = pd.to_datetime("2011-12-01")
# end_date = pd.to_datetime("2013-07-01")
#
# while end_date < pd.to_datetime("2021-03-01"):
#     train_period_list.append((start_date-relativedelta(days=1), end_date-relativedelta(days=1)))
#     start_date += relativedelta(months=1)
#     end_date += relativedelta(months=1)
#
# raw_data = pd.read_pickle(f"./raw_factor_data_std/raw_factor_data_std.pkl")
# for start_dt, end_dt in train_period_list:
#     real_end_dt,end_dt_str = get_last_end_trade_date(start_dt, end_dt)
#     print(f"Create training data set from {start_dt} to {real_end_dt} ....")
#     required_years = list({str(start_dt)[:4], str(end_dt)[:4]})
#     print("Load data...")
#     required_data = raw_data[(raw_data['date']>=start_dt) & (raw_data['date']<=real_end_dt)]
#     required_data.to_pickle(f"./data_for_training/{end_dt_str}.pkl")
#     print("Finish!")

"""#########################################"""

"""Normalize the data"""
# train_period_list = []
# for i in range(2012, 2022):
#     train_period_list.append((pd.to_datetime(str(i)+"0101"), pd.to_datetime(str(i)+"1231")))
#
# no_norm_cols = ['code', 'date', 'open', 'high', 'low','close', 'pre_close', 'ref']
# for start_date, end_date in train_period_list:
#     print(f"From {start_date} to {end_date}....")
#     output = pd.DataFrame()
#     print("Start loading data....")
#     for i in range(1,6):
#         data = pd.read_pickle(f"./raw_stock_factors/raw_factor_data_{i}.pkl")
#         data = data[(data['date']>=start_date) & (data['date']<=end_date)]
#         output = pd.concat([output, data.copy(deep=True)], axis=0)
#     print("Finish loading data!")
#     factor_cols = [f for f in output.columns if f not in no_norm_cols]
#     norm_output = pd.DataFrame()
#     print("Start normalizing data....")
#     for date, date_data in output.groupby("date"):
#         date_data.loc[:, factor_cols] = date_data.loc[:, factor_cols].apply(lambda x: (x - x.mean()) / (x.std()))
#         norm_output = pd.concat([norm_output, date_data])
#     print(f"Finish normalizing {str(start_date)[:4]} data!\n")
#     norm_output.to_pickle(f"./yearly_norm_raw_stock_factors/{str(start_date)[:4]}.pkl")
"""#################################################################################"""

"""Load market cap"""
# factor_data_csv = os.listdir("./nero_factors_data - 副本/")
# market_data = pd.read_csv(f"./data/raw_daily_indicator_data.csv", index_col=0)
# market_data.rename(columns={'trade_date':'date','ts_code':'code'}, inplace=True)
# market_data = market_data.loc[:, ["code", "date", "total_mv"]]
# market_data['date'] = [str(date)[:8] for date in market_data['date'].tolist()]
# market_data['date'] = pd.to_datetime(market_data['date'])
# for csv in factor_data_csv:
#     print(f"Process {csv}...")
#     factor_data = pd.read_csv(f"./nero_factors_data - 副本/{csv}", index_col=0)
#     factor_data['date'] = pd.to_datetime(factor_data['date'])
#     print(factor_data.head())
#     print(market_data.head())
#     outputs = pd.merge(left=factor_data, right=market_data, on=["code","date"], how="inner")
#     outputs.to_csv(f"./nero_factors_data/{csv}")
"""##########################################################"""

"""NaN data and extreme data"""
# from data_process import Data_Process
# import time
# import pandas as pd
# import os
# from multiprocessing import Pool
# import numpy as np
# from copy import deepcopy
# s = time.time()
# factor_csv = os.listdir("./factor_data_csv/")
# dh = Data_Process()
# # data = pd.read_pickle("./raw_factor_data/raw_factor_data_old.pkl")
# data = pd.read_csv("./data/raw_daily_indicator_data.csv", index_col=0)
# data.rename(columns={"ts_code":"code", "trade_date":"date"},inplace=True)
#
# data["date"] = [pd.to_datetime(str(dt)) for dt in data["date"].to_list()]
# data["code"] = data["code"].apply(lambda x:x[:6])
#
# def get_std_data3(date):
#     print(f"Process data on {date}...")
#     code_list = dh.get_ZZ800_components(date)
#     date_factor_data = data[(data["date"]==pd.to_datetime(date)) & (data["code"].isin(code_list))]
#     col = "total_mv"
#     col_indx = date_factor_data.columns.get_loc(col)
#     date_factor_data = date_factor_data.iloc[:,[0,1,col_indx]]
#     one_factor_data = date_factor_data[col]
#     F_M = one_factor_data.median()
#     F_M1 = abs(one_factor_data - F_M).median()
#     one_factor_data.fillna(value=F_M)
#     one_factor_data_without_em = one_factor_data.apply(lambda x: F_M-5*F_M1 if x < F_M-5*F_M1 \
#                                                             else (x if x < F_M+5*F_M1 else F_M+5*F_M1))
#     one_factor_data_without_em = (one_factor_data_without_em - one_factor_data_without_em.mean()) / one_factor_data_without_em.std()
#     date_factor_data[col] = one_factor_data_without_em
#     return date_factor_data
#
# def get_std_data2(date):
#     print(f"Process data on {date}...")
#     code_list = dh.get_ZZ800_components(date)
#     date_factor_data = data[(data["date"]==pd.to_datetime(date)) & (data["code"].isin(code_list))]
#     F_M = date_factor_data.iloc[:, 2:-1].median()
#     F_M1 = date_factor_data.iloc[:, 2:-1].apply(lambda x:abs(x-x.median()), axis=0).median()
#     for col in date_factor_data.columns[2:-1]:
#         one_factor_data = date_factor_data[col]
#         one_factor_data.fillna(value=F_M[col])
#         one_factor_data_without_em = one_factor_data.apply(lambda x: F_M[col]-5*F_M1[col] if x < F_M[col]-5*F_M1[col] \
#                                                                 else (x if x < F_M[col]+5*F_M1[col] else F_M[col]+5*F_M1[col]))
#         one_factor_data_without_em = (one_factor_data_without_em - one_factor_data_without_em.mean()) / one_factor_data_without_em.std()
#         date_factor_data[col] = one_factor_data_without_em
#     return date_factor_data
#
# # for csv in factor_csv:
# def get_std_data(csv):
#     print(f"Process {csv}...")
#     factor_data = pd.read_csv(f"./factor_data_csv/{csv}", index_col=1)
#     factor_name = csv.replace(".csv", "")
#     factor_data.columns = [stock_id[:6] for stock_id in factor_data.columns]
#     factor_data.drop(columns=factor_data.columns[0:1],inplace=True)
#     one_factor_data = pd.DataFrame(columns=["code","date",factor_name])
#     i = 0
#     for date in factor_data.index[1:]:
#         if i%100==0:
#             print(date)
#         i += 1
#
#         code_list = dh.get_ZZ800_components(date)
#         date_factor_data = factor_data.loc[date, code_list]
#         F_M = date_factor_data.median()
#         F_M1 = (abs(date_factor_data - F_M)).median()
#         date_factor_data.fillna(value=F_M, inplace=True)
#         date_factor_data_without_extreme = date_factor_data.apply(lambda x: F_M-5*F_M1 if x < F_M-5*F_M1 else (x if x < F_M+5*F_M1 else F_M+5*F_M1))
#         date_factor_data_without_extreme = (date_factor_data_without_extreme - date_factor_data_without_extreme.mean()) / date_factor_data_without_extreme.std()
#         one_factor_data = pd.concat([one_factor_data, pd.DataFrame({"code":date_factor_data_without_extreme.index,
#                                 "date":[date]*len(date_factor_data),
#                                 factor_name:date_factor_data_without_extreme.to_list()})],axis=0)
#     return one_factor_data
#
# def split_list_to_n_parts(_list, n):
#     for i in range(0, len(_list), int(np.ceil(len(_list)/n))):
#         yield _list[i:i+int(np.ceil(len(_list)/n))]
#
# if __name__ == "__main__":
#     pool = Pool(8)
#     all_dates = list(set(data["date"]))
#     all_dates.sort()
#     all_dates = all_dates[1:]
#     date_group = split_list_to_n_parts(all_dates, 10)
#     output = pd.DataFrame()
#     while True:
#         try:
#             date_list = next(date_group)
#             date_factor_data_tmp = pool.map(get_std_data3, date_list)
#             for date_factor_data in date_factor_data_tmp:
#                 output = pd.concat([output, date_factor_data], axis=0)
#         except StopIteration:
#             break
#     pool.close()
#     pool.join()
#     output.to_pickle("./raw_factor_data_std/raw_market_value_std.pkl")
    # pool = Pool(30)
    # csv_group = split_list_to_n_parts(factor_csv, 5)
    # i = 1
    # while True:
    #     try:
    #         csv_list = next(csv_group)
    #         print(f"start group {csv_list}")
    #         all_factor_data = pool.map(get_std_data, csv_list)
    #         combined_df = pd.DataFrame()
    #         for one_factor_data in all_factor_data:
    #             if combined_df.empty:
    #                 combined_df = deepcopy(one_factor_data)
    #             else:
    #                 combined_df = pd.merge(combined_df, one_factor_data, how="outer", on=["code", "date"])
    #         combined_df.to_pickle(f"./raw_factor_data_std/raw_financial_factor_data_std_{i}.pkl")
    #         del all_factor_data
    #         del combined_df
    #         print(f"###################### Finish {i}-th group ####################")
    #         i += 1
    #     except StopIteration:
    #         break
    # pool.close()
    # pool.join()
"""########################################################################################"""

"""合并标准化后的因子数据"""
# import os
# import pandas as pd
#
# pkls = os.listdir("./raw_factor_data_std/")
# output = pd.DataFrame()
# for pkl in pkls:
#     print(f"Process {pkl}...")
#     data_tmp = pd.read_pickle(f"./raw_factor_data_std/{pkl}")
#     data_tmp["date"] = pd.to_datetime(data_tmp["date"])
#     data_tmp["code"] = data_tmp["code"].apply(lambda x:x[:6])
#     if output.empty:
#         output = data_tmp
#     else:
#         output = pd.merge(output, data_tmp, on=["code", "date"])
# output.iloc[:,2:-1].dropna(axis=1, inplace=True)
# output.to_pickle("./raw_factor_data_std/raw_factor_data_std.pkl")

"""Yield yearly raw factor data"""
# import pandas as pd
# data = pd.read_pickle("./raw_factor_data_std/raw_factor_data_std.pkl")
# for year in range(2013, 2022):
#     start_date = pd.to_datetime(str(year)+"-01-01")
#     end_date = pd.to_datetime(str(year)+"-12-31")
#     print(f"Load data from {start_date} to {end_date}")
#     yearly_data = data.loc[(data["date"]>=start_date) & (data["date"]<=end_date)]
#     yearly_data.to_pickle(f"./yearly_norm_raw_stock_factors/{year}.pkl")

"""Rename nero factor model"""
# from parameters import FACTOR_CATEGORY
# import os
# import pickle
# from train_nero_factors import my_sigmoid,my_activation
# factor_names = FACTOR_CATEGORY.keys()
# nero_factor_net_pkls = [net for net in os.listdir("./nero_factors_model/") if "winner" not in net]
# for factor in factor_names:
#     factor_net_pkls = [net for net in nero_factor_net_pkls if factor in net]
#     factor_net_pkls.append(f"{factor}_20210228.pkl")
#     factor_net_pkls.sort(key=lambda x: int(x[-12:-4]), reverse=True)
#     for pkl, next_pkl in zip(factor_net_pkls[1:], factor_net_pkls[:-1]):
#         model = pickle.load(open(f"./nero_factors_model/{pkl}", "rb"))
#         file = open(f"./nero_factors_model_new/{next_pkl}", "wb")
#         pickle.dump(model, file)
#         file.close()

"""Reduce dimension"""
# import pandas as pd
# from parameters import FACTOR_CATEGORY
# factor_names = list(FACTOR_CATEGORY.keys())
# for factor in factor_names:
#     print(f"{factor}:{len(FACTOR_CATEGORY[factor])}")
# data = pd.read_pickle("./raw_factor_data_std/raw_factor_data_std.pkl")
# cor = data.loc[:, FACTOR_CATEGORY[factor_names[6]]].corr()
# original_num = len(FACTOR_CATEGORY[factor_names[6]])

# for i,col in enumerate(cor.columns):
#     count = 0
#     print()
#     for indx in cor.index:
#         if cor.at[indx,col] > 0.7 and indx != col:
#             count += 1
#             print(col,indx,cor.at[indx,col])
#     print(f"{col} correlation num:{count}")


"""将计算的nero factor与市值跟收盘价合并"""
import os
import pandas as pd
all_csv = [file for file in os.listdir("./nero_factors_data/") if "csv" in file]
all_stock_data = pd.DataFrame()
for csv in all_csv:
    print(f"Process {csv}")
    yearly_factor_data = pd.read_csv(f"./nero_factors_data/{csv}", index_col=0)
    all_stock_data = pd.concat([all_stock_data, yearly_factor_data], axis=0)
all_stock_data["date"] = pd.to_datetime(all_stock_data["date"])
stock_market_data = pd.read_csv("./data/raw_market_data.csv", index_col=0)
market_cap = pd.read_csv("./data/raw_daily_indicator_data.csv", index_col=0)
stock_market_data = stock_market_data.loc[:, ["ts_code", "trade_date", "close"]]
market_cap = market_cap.loc[:, ["ts_code", "trade_date", "total_mv"]]
stock_market_data = pd.merge(stock_market_data, market_cap, how="left", on=["ts_code", "trade_date"])
stock_market_data.rename(columns={"ts_code": "code", "trade_date": "date"}, inplace=True)
stock_market_data["date"] = [pd.to_datetime(str(dt)) for dt in stock_market_data["date"].tolist()]
all_stock_data = pd.merge(stock_market_data,all_stock_data,how="left", on=["code", "date"])
all_stock_data.to_pickle("./nero_factors_data/all_factor_data.pkl")