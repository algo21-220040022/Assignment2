import pandas as pd
import numpy as np

class Data_Process(object):
    def __init__(self):
        self.ZZ800_components_df = pd.read_csv("./data/daily_components110301-210301.csv", index_col=0)
        self.ZZ800_components_df.columns = pd.to_datetime(self.ZZ800_components_df.columns)
        self.is_load_raw_data = False

    def set_factor_list(self, factor_list):
        self.factor_list = factor_list


    def load_training_data(self, file):
        print(f"Load data from {file}")
        self.raw_factor_data = pd.read_pickle(file)
        #self.raw_factor_data.fillna(0, inplace=True)
        self.is_load_raw_data = True
        self.raw_factor_data.index = self.raw_factor_data['date']

    def get_ZZ800_components(self, date):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        ZZ800_date = self.ZZ800_components_df.columns[self.ZZ800_components_df.columns < date][-1]
        ZZ800_components = [code[:6] for code in self.ZZ800_components_df[ZZ800_date].tolist()]
        if self.is_load_raw_data:
            has_factor_stocks = list(set(self.raw_factor_data[self.raw_factor_data['date']==date]['code']))
            return [c for c in has_factor_stocks if c[:6] in ZZ800_components]
        else:
            return ZZ800_components

    def get_network_input_and_ret_f(self, stock_id, date):
        """get the input for network and the future return"""
        # if isinstance(date, str):
        #     date = pd.to_datetime(date)
        date_data = self.raw_factor_data[self.raw_factor_data['code'] == stock_id].loc[date,:]
        return np.array(date_data[self.factor_list]), date_data['ret_f']

    def get_all_network_inputs_and_ret_f(self, stock_ids, date):
        date_data = self.raw_factor_data[(self.raw_factor_data['date']==date) & (self.raw_factor_data['code'].isin(stock_ids))]
        factor_data_df = date_data.loc[:, self.factor_list+['ret_f']]
        factor_data_df.dropna(axis=0, inplace=True)
        all_network_inputs = np.array(factor_data_df.loc[:, self.factor_list])
        ret_f_arr = np.array(factor_data_df['ret_f'])
        return all_network_inputs, ret_f_arr

    def get_all_network_inputs(self, date):
        date_data = self.raw_factor_data[self.raw_factor_data['date'] == date]
        factor_data_df = date_data.loc[:, ['code', 'date']+self.factor_list]
        factor_data_df.dropna(axis=0, inplace=True)
        all_network_inputs = np.array(factor_data_df.loc[:, self.factor_list])
        return factor_data_df['code'].tolist(), factor_data_df['date'].tolist(), all_network_inputs

    def get_all_trade_dates(self):
        all_trade_dates = list(set(self.raw_factor_data['date']))
        all_trade_dates.sort()
        return all_trade_dates



if __name__ == "__main__":
    """Just for test"""
    from parameters import FACTOR_CATEGORY
    data_handler = Data_Process()
    print(data_handler.get_all_network_inputs_and_ret_f(['000001.SZ','000002.SZ'], date=pd.to_datetime("2012-06-04")))
