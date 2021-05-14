import os
import numpy as np
import pandas as pd
import talib
from parameters import PREDICT_PERIOD
from parameters import FACTOR_CATEGORY
from data_process import Data_Process
import pickle
from pathos.multiprocessing import ProcessingPool as newPool
import traceback
from dateutil.relativedelta import relativedelta


data_handler = Data_Process()
code_dict = pickle.load(open("./code_dict.pkl","rb"))
def my_sigmoid(x):
    return 1/(1+np.exp(-0.5*x))

def my_activation(x):
    return (x)/(10e-10+10*abs(x/30)**(1/3))


def generate_yearly_nero_factor_data(year, factor_names):
    print(f"########## Start generating data in {year} ##########")
    try:
        outputs = pd.read_csv(f"./nero_factors_data/{year}.csv", index_col=0)
        outputs['date'] = pd.to_datetime(outputs['date'])
        #outputs["code"] = Generate_Factor_Data.format_codes(outputs["code"].to_list())
        print(outputs.head())
    except:
        traceback.print_exc()
        outputs = pd.DataFrame()
    all_yearly_factor_nets = Generate_Factor_Data.get_yearly_factor_nets(factor_names, year)
    data_handler.load_training_data(file=f"./yearly_norm_raw_stock_factors/{year}.pkl")
    for factor_name, factor_nets in all_yearly_factor_nets.items():
        print(f"Generate data of factor [{factor_name}]....")
        data_handler.set_factor_list(FACTOR_CATEGORY[factor_name])
        nero_factor_data = []
        code_list = []
        date_list = []
        for i, net in enumerate(factor_nets[:-1]):
            start_date = net[0]
            end_date = factor_nets[i + 1][0]
            print(f"Data from {start_date} to {end_date} ")
            all_trade_dates = [date for date in data_handler.get_all_trade_dates() if
                               (date >= start_date) and (date < end_date)]
            codes_dates_network_inputs = map(data_handler.get_all_network_inputs, all_trade_dates)
            print(f"Complete create network inputs...")
            for codes, dates, network_inputs in codes_dates_network_inputs:
                print(f"Generate data on {dates[0]}...")
                code_list.extend(codes)
                date_list.extend(dates)
                nero_factor_data.extend(list(map(lambda x: net[1].activate(x)[0], network_inputs)))
        code_list = Generate_Factor_Data.format_codes(code_list)
        date_list = [pd.to_datetime(dt) for dt in date_list]
        nero_factor_df = pd.DataFrame({"code": code_list, "date": date_list, factor_name: nero_factor_data})
        if outputs.empty:
            outputs = nero_factor_df
        else:
            print(f"Result :\n{nero_factor_df.head()}")
            outputs = pd.merge(left=outputs, right=nero_factor_df, on=["code", "date"])
    outputs.to_csv(f"./nero_factors_data/{year}.csv")
    print(f"########## Finish generating data in {year} ##########\n")

class Generate_Factor_Data(object):
    def __init__(self):
        self.nero_factors_pkls = os.listdir("./new_factors/")
        self.all_train_end_dates = list(set([pkl[-14:-4] for pkl in self.nero_factors_pkls]))
        self.nero_factor_names = [pkl[:-15] for pkl in self.nero_factors_pkls]
        self.stock_id_list = []
        self.all_factor_data = {}

    @staticmethod
    def generate_basic_factor_data():
        """Generate all basic factors"""

        raw_daily_data = pd.read_pickle("./data/raw_daily_data.pkl")
        raw_financial_data = pd.read_pickle("./data/raw_financial_data.pkl")
        raw_daily_data['date'] = pd.to_datetime(raw_daily_data['date'])
        raw_financial_data['ann_date'] = pd.to_datetime(raw_financial_data['ann_date'])

        financial_factor_names = list(raw_financial_data.columns)[3:]

        stock_groups = raw_daily_data.groupby(by="code")

        raw_factors_pkls = os.listdir("./raw_stock_factors/")
        OK_stock_ids = [pkl[:6] for pkl in raw_factors_pkls]
        print(f"OK Stock: {OK_stock_ids}")
        for stock_id, stock_data in stock_groups:
            if stock_id[:6] in OK_stock_ids:
                continue
            print(f"Process {stock_id} .....")

            stock_data.sort_values(by='date', ascending=True, inplace=True)
            high = stock_data['high']
            low = stock_data['low']
            open = stock_data['open']
            close = stock_data['close']
            volume = stock_data['vol']
            indicators_df = stock_data
            drop_col_start = indicators_df.shape[1]
            indicators_df.loc[:,'daily_ret'] = indicators_df['close'].pct_change()

            """momentum indicators"""
            indicators_df.loc[:,'ADX'] = talib.ADX(high, low, close, timeperiod=14)  # Average Directional Movement Index
            indicators_df.loc[:,'ADXR'] = talib.ADXR(high, low, close, timeperiod=14)  # Average Directional Movement Index Rating
            indicators_df.loc[:,'APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)  # Absolute Price Oscillator
            indicators_df.loc[:,'aroondown'], indicators_df.loc[:,'aroonup'] = talib.AROON(high, low, timeperiod=14)  # Aroon
            indicators_df.loc[:,'AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)  # Aroon Oscillator
            indicators_df.loc[:,'BOP'] = talib.BOP(open, high, low, close)  # Balance Of Power
            indicators_df.loc[:,'CCI'] = talib.CCI(high, low, close, timeperiod=14)  # Commodity Channel Index
            indicators_df.loc[:,'CMO'] = talib.CMO(close, timeperiod=14)  # Chande Momentum Oscillator
            indicators_df.loc[:,'DX'] = talib.DX(high, low, close, timeperiod=14)  # Directional Movement Index
            indicators_df.loc[:,'macd1'], indicators_df.loc[:,'macdsignal1'], macdhist = talib.MACD(close, fastperiod=12,
                                                                                        slowperiod=26,signalperiod=9)  # Moving Average Convergence/Divergence
            indicators_df.loc[:,'MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)  # Money Flow Index
            indicators_df.loc[:,'MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)  # Minus Directional Indicator
            indicators_df.loc[:,'MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)  # Minus Directional Movement
            indicators_df.loc[:,'MOM'] = talib.MOM(close, timeperiod=10)  # Momentum
            indicators_df.loc[:,'PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)  # Plus Directional Indicator
            indicators_df.loc[:,'PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)  # Plus Directional Movement
            indicators_df.loc[:,'PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26,matype=0)  # Percentage Price Oscillator
            indicators_df.loc[:,'RSI'] = talib.RSI(close, timeperiod=14)  # Relative Strength Index
            indicators_df.loc[:,'S-RSIk'], indicators_df.loc[:,'S-RSId'] = talib.STOCHRSI(close, timeperiod=20, fastk_period=10,
                                                                              fastd_period=5, fastd_matype=0)  # Stochastic Relative Strength Index
            indicators_df.loc[:,'TRIX'] = talib.TRIX(close, timeperiod=30)  # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
            indicators_df.loc[:,'ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)  # Ultimate Oscillator
            indicators_df.loc[:,'WILLR'] = talib.WILLR(high, low, close, timeperiod=14)  # Williams' %R

            """Volume indicators"""
            indicators_df.loc[:,'AD'] = talib.AD(high, low, close, volume)  # Chaikin A/D Line
            indicators_df.loc[:,'ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=10, slowperiod=20)  # Chaikin A/D Oscillator
            indicators_df.loc[:,'OBV'] = talib.OBV(close, volume)  # On Balance Volume
            indicators_df.loc[:,'volume_pct'] = indicators_df['vol'].pct_change(periods=PREDICT_PERIOD)
            indicators_df.loc[:,'volume_std'] = indicators_df['vol'].rolling(window=20).std()

            """Volatility indicators"""
            indicators_df.loc[:,'ATR'] = talib.ATR(high, low, close, timeperiod=20)  # Average True Range
            indicators_df.loc[:,'NATR'] = talib.NATR(high, low, close, timeperiod=20)  # Normalized Average True Range
            indicators_df.loc[:,'TRANGE_1'] = talib.TRANGE(high, low, close).rolling(window=PREDICT_PERIOD).mean()  # True Range
            indicators_df.loc[:,'TRANGE_2'] = talib.TRANGE(high, low, close).rolling(window=PREDICT_PERIOD*3).mean()
            indicators_df.loc[:,'TRANGE_3'] = talib.TRANGE(high, low, close).rolling(window=PREDICT_PERIOD*6).mean()
            indicators_df.loc[:,'TRANGE_4'] = talib.TRANGE(high, low, close).rolling(window=PREDICT_PERIOD*12).mean()
            indicators_df.loc[:,'volatility_1'] = indicators_df['daily_ret'].rolling(window=20).std()
            indicators_df.loc[:,'volatility_2'] = indicators_df['daily_ret'].rolling(window=20 * 3).std()
            indicators_df.loc[:,'volatility_3'] = indicators_df['daily_ret'].rolling(window=20 * 6).std()
            indicators_df.loc[:,'volatility_4'] = indicators_df['daily_ret'].rolling(window=20 * 12).std()

            """Lag return"""
            indicators_df.loc[:,'ret'] = indicators_df['close'].pct_change(periods=PREDICT_PERIOD)
            indicators_df.loc[:,'ret_lag1'] = indicators_df['close'].shift(periods=PREDICT_PERIOD).pct_change(periods=PREDICT_PERIOD)
            indicators_df.loc[:,'ret_lag2'] = indicators_df['close'].shift(periods=PREDICT_PERIOD*2).pct_change(periods=PREDICT_PERIOD)

            # drop null rows
            indicators_df.dropna(axis=0, subset=indicators_df.columns[drop_col_start:], inplace=True)

            """Financial indicators"""
            one_stock_financial_data = raw_financial_data[raw_financial_data['code']==stock_id]
            one_stock_financial_data.sort_values(by="ann_date", inplace=True)
            one_stock_financial_data.index = range(one_stock_financial_data.shape[0])
            for name in financial_factor_names:
                indicators_df.insert(indicators_df.shape[1], name, np.zeros(indicators_df.shape[0]))

            print("Process financial indicators")
            daily_dates = np.array(indicators_df['date'].tolist())
            if len(daily_dates) == 0:
                print(f"No enough data for {stock_id} ......")
                continue

            for factor_name in financial_factor_names:
                #factor_name = "eps"
                factor_list = []
                ann_date_arr = np.array(one_stock_financial_data['ann_date'].tolist()+[pd.to_datetime("2030-01-01")])
                nan_num = np.argmin(daily_dates <= ann_date_arr[0])
                factor_list.extend([np.nan]*nan_num)
                for i,date in enumerate(ann_date_arr[:-1]):
                    print(f"{factor_name}:{date}...")
                    try:
                        start_indx = np.where(daily_dates > date)[0][0]
                        end_indx = np.where(daily_dates <= ann_date_arr[i+1])[0][-1] + 1
                    except IndexError:
                        start_indx = 0
                        end_indx = 0
                    factor_value = one_stock_financial_data.at[i, factor_name]
                    factor_list.extend([factor_value]*(end_indx-start_indx))

                indicators_df.loc[:, factor_name] = factor_list

            """Future return"""
            indicators_df.loc[:,'ret_f'] = (indicators_df['close'].shift(periods=-PREDICT_PERIOD) -
                                      indicators_df['open'].shift(periods=-1)) / indicators_df['open'].shift(periods=-1)

            print(f'Output the factor data of {stock_id} to .pkl file.....')
            indicators_df.to_pickle(f"./raw_stock_factors/{stock_id[:6]}.pkl")
            print(f'{stock_id} finish!\n')

    @staticmethod
    def get_yearly_factor_nets(factor_names, year):
        nero_factor_net_pkls = [net for net in os.listdir("./nero_factors_model/") if "winner" not in net]
        factor_net_dict = {}
        for factor in factor_names:
            factor_net_pkls = [net for net in nero_factor_net_pkls if (factor in net) and (int(net[-12:-8])==year)]
            date_list = []
            factor_net_list = []
            if year != 2013:
                last_year_net_pkls = [net for net in nero_factor_net_pkls if (factor in net) and (int(net[-12:-8])==year-1)]
                last_year_net_pkls.sort(key=lambda x:int(x[-12:-4]))
                last_year_end_net = pickle.load(open(f"./nero_factors_model/{last_year_net_pkls[-1]}", "rb"))
                date_list.append(pd.to_datetime(last_year_net_pkls[-1][-12:-4]))
                factor_net_list.append(last_year_end_net)
            for net in factor_net_pkls:
                factor_net = pickle.load(open(f"./nero_factors_model/{net}", "rb"))
                factor_net_list.append(factor_net)
            date_list.extend([pd.to_datetime(pkl[-12:-4]) for pkl in factor_net_pkls])
            factor_net_tmp = list(zip(date_list, factor_net_list))
            factor_net_tmp.pop(-1)
            factor_net_tmp.append((pd.to_datetime("20300101"), None))
            factor_net_tmp.sort(key=lambda x: x[0])
            factor_net_dict[factor] = factor_net_tmp
        return factor_net_dict

    @staticmethod
    def format_codes(code_list):
        return [code_dict[int(c)] for c in code_list]

    @staticmethod
    def generate_nero_factor_data(factor_names = ["momentum"]):

        years = list(range(2013, 2022))
        years = [2014]
        pool = newPool(len(years))
        pool.map(generate_yearly_nero_factor_data, years, [factor_names]*len(years))
        pool.close()
        pool.join()
        # generate_yearly_nero_factor_data(2014, factor_names)
        # for year in years:



def main():
    #Generate_Factor_Data.generate_basic_factor_data()
    target_factors = list(FACTOR_CATEGORY.keys())
    target_factors.remove("composite_factor")
    # target_factors.remove("profitability")
    # target_factors.remove("growth")
    # target_factors.remove("levarage")
    # print(f"Target factors: growth")
    Generate_Factor_Data.generate_nero_factor_data(factor_names=["composite_factor"])

if __name__ == "__main__":
    main()