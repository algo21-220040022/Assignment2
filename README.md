# Neural_Network_Factor_Mining

This project utilizes the evolutionary neural network to mine some new factors from general factors.
The evolutionary neural network employs the genetic algorithm idea to train the network. Here the objective of the training is the Rank IC of the new factor. So the aim of this method is to find the best neural network structure (that is, with highest rank IC) to generate new factors.

## backtest_result
This folder is used to store the backtest results.

## config
The config files required in evolutionary neural network are in this folder.

## data
It contains the OHLCV daily price data of CSI800.

## data_for_training
This folder is used to store the data used in evolutionary neural network training.

## factor_data_csv
Some financial raw indicator data such as ROA, ROE is stored in this folder.

## nero_factors_data
The new factors trained from evolutionary neural network are stored here.

## nero_factor_model
The models trained from evolutionary neural network are stored here.

## raw_factor_data_std
This folder is used to store the data which is standardized and de-exetreme-value.

## yearly_norm_raw_stock_factors
This folder is used to store the yearly raw factor data. In my old computer, the whole raw data file is too big to load in Python, so I divide it into yearly file. While it is not neccesary in my new computer for it has larger RAM.

## backtest.py
It conduct the backtest process and output the result to backtest_result folder.

## data_process.py
This file define an data handler object to process the data during the training.

## dirty_work.py
This file do some data preparation jobs including standardization, filter exetreme value, combine different data and etc.

## train_nero_factors.py
This file is the main file used to train the evolutionary neural network.

## fitness_calculation.py
This file is used to compute the fitness value of evolutionary neural network.

## generate_factor_data_for_backtest.py
This file is used to generate both some technical data and the final neural network factor data.

## parameters.py
This file is used to define the categories of different factors and other parameters.

## backtest result
![image](https://github.com/algo21-220040022/Neural_Network_Factor_Mining/blob/main/picture/backtest_reuslt.png.png)
