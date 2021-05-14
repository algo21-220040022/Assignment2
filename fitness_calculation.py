"""fitness_calculation.py"""
"""calculate fitness value (Rank IC) given a factor"""
from parameters import W_MEAN, W_STD, FACTOR_CATEGORY
import numpy as np
#from multiprocessing import Pool, Manager
from scipy.stats import spearmanr

def fitness_cal(net, data_handler):

    all_trade_dates = data_handler.get_all_trade_dates()
    sampled_trade_dates = [np.random.choice(sub_set, 1)[0] for sub_set in np.array_split(np.array(all_trade_dates),50)]
    num_date = len(sampled_trade_dates)

    """Map process"""

    def cal_rank_IC_mp(date):
        stock_ids = data_handler.get_ZZ800_components(date)
        # factor_arr = np.zeros(len(stock_ids))
        all_stock_network_inputs, stock_ret_arr = data_handler.get_all_network_inputs_and_ret_f(stock_ids, date)
        factor_arr = np.array(list(map(lambda x: net.activate(x)[0], all_stock_network_inputs)))
        # for i,network_input in enumerate(all_stock_network_inputs):
        #     factor_arr[i] = net.activate(network_input)[0]
        # print(f"Raw data: {all_stock_network_inputs[range(0,len(stock_ids),100)]}")
        print(f"Factor values: {data_handler.factor_list[0]}{factor_arr[range(0, factor_arr.shape[0], 100)]}")
        if abs(sum(factor_arr)) < 0.1:
            print(all_stock_network_inputs[range(0, factor_arr.shape[0], 100),:])
        rank_IC = spearmanr(factor_arr, stock_ret_arr)[0]
        return rank_IC
    # pool = newPool(6)
    rank_IC_arr = list(map(cal_rank_IC_mp, sampled_trade_dates))
    # pool.close()
    # pool.join()
    """Old version"""
    # rank_IC_arr = np.zeros(num_date)
    # for i,date in enumerate(sampled_trade_dates):
    #     rank_IC_arr[i] = cal_rank_IC(date, net, data_handler)
    #     if (i+1) % 10 == 0:
    #         print(f"Single Rank IC: {rank_IC_arr[i]}")
    rank_IC_median = np.median(rank_IC_arr)
    rank_IC_std = np.std(rank_IC_arr)
    fitness = W_MEAN*rank_IC_median - W_STD*rank_IC_std
    print(f"Median Rank IC(fitness): {rank_IC_median}")
    return fitness


def cal_rank_IC(date, net, data_handler):
    stock_ids = data_handler.get_ZZ800_components(date)
    # factor_arr = np.zeros(len(stock_ids))
    all_stock_network_inputs, stock_ret_arr = data_handler.get_all_network_inputs_and_ret_f(stock_ids, date)
    factor_arr = np.array(list(map(lambda x: net.activate(x)[0], all_stock_network_inputs)))
    # for i,network_input in enumerate(all_stock_network_inputs):
    #     factor_arr[i] = net.activate(network_input)[0]
    #print(f"Raw data: {all_stock_network_inputs[range(0,len(stock_ids),100)]}")
    print(f"Factor values: {factor_arr[range(0,factor_arr.shape[0],100)]}")
    rank_IC = spearmanr(factor_arr, stock_ret_arr)[0]
    return rank_IC
