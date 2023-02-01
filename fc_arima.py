import numpy as np
import pandas as pd
import psutil as ps
from joblib import Parallel
from joblib import delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsforecast.arima import auto_arima_f
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer

from output_logger import Logger
import sys, os
from util_funcs import *
from collections import Counter



pd.set_option('display.max_columns', None)



class ARMA_opt:
	def __init__(self, data, n_test, pqs=None, norm=True):
		# 初始化保存所需数据
		self.data = data
		self.norm = norm
		self.fc_size = n_test
		self.train, self.test = self.train_test_split()
		self.max_order = self.get_max_order()
		self.max_p = self.max_q = self.max_order + 100  # 尽量加大避免边界问题
		self.best_pq = self.auto_pq() if pqs is None else pqs
		self.fc_rmse, self.fc_nrmse, self.fc_mae, self.fc_r2 = self.auto_score()

	# split a multivariate dataset into train/test sets
	def train_test_split(self):
		return self.data[:-self.fc_size], self.data[-self.fc_size:]

	def get_max_order(self):
		orders = []
		for i in self.data.columns:
			print(f'get period for : {i}')
			orders += set(get_period_welch(self.train[i].dropna()))
		orders = sorted(Counter(orders).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		return orders[0][0]

	def _pq_seq1(self, name):
		print(f'Searching order of p and q for : {name}')
		stepwise_model = auto_arima(self.train[name].dropna(), start_p=0, start_q=0,
		                            max_p=self.max_p, max_q=self.max_q, d=0, seasonal=False,
		                            trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
		                            information_criterion ='hqic', maxiter=500)
		parameter = stepwise_model.get_params().get('order')
		print(f'optimal order for:{name} is: {parameter} \n\n')
		return name, parameter

	def _pq_seq(self, name):
		print(f'Searching order of p and q for : {name}')
		if self.norm:
			scaler = RobustScaler()
			data = scaler.fit_transform(self.train[name].dropna().values.reshape(-1, 1)).flatten()
		else:
			data = self.train[name].dropna().values
		stepwise_model = auto_arima_f(x=data, start_p=0, start_q=0,
		                              max_p=self.max_p, max_q=self.max_q, d=0,
		                              seasonal=False, num_cores=4,
		                              stepwise=True, trace=True, ic='bic')
		p, q, P, Q, s, d, D = stepwise_model.get('arma')
		order = (p, d, q)
		# seasonal_order = (P, D, Q, s)
		if stepwise_model.get('coef').get('intercept', None) is not None:
			trend = 'c'
		else: trend = 'n'
		print(f'optimal order for:{name} is: {order} with trend={trend} \n\n')
		return name, (order, trend)

	def auto_pq(self, parallel=True):
		pq = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 28) * 0.4
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self._pq_seq)(name=name) for name in self.train.columns)
			res = executor(tasks)
			pq = dict(res)
		else:
			for name in self.train.columns:
				pq[name] = self._pq_seq(name)[1]
		print(f'optimal all data orders for arima is: {pq} \n\n')
		return pq

	def _fc_score(self, name):
		order, trend = self.best_pq[name]
		if self.norm:
			scaler = RobustScaler()
			data = scaler.fit_transform(self.train[name].dropna().values.reshape(-1, 1))
		else:
			data = self.train[name].dropna()
		model = SARIMAX(data, order=order, trend=trend).fit(maxiter=np.inf, method='powell', maxfun=np.inf)
		if self.norm:
			result = scaler.inverse_transform(model.forecast(steps=self.fc_size).reshape(-1, 1))
		else:
			result = model.forecast(steps=self.fc_size)
		result = model.forecast(steps=self.fc_size)
		rmse = np.sqrt(mean_squared_error(self.test[name], result))
		nrmse = rmse/np.mean(self.test[name])
		mae = mean_absolute_error(self.test[name], result)
		r2 = r2_score(self.test[name], result)
		print(name, rmse)
		return name, rmse, nrmse, mae, r2

	def auto_score(self, parallel=True):
		rmse, nrmse, mae, r2 = {}, {}, {}, {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 28) * 0.4
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self._fc_score)(name=name) for name in self.train.columns)
			res = executor(tasks)
			for item in res:
				rmse[item[0]], nrmse[item[0]], mae[item[0]], r2[item[0]] = item[1], item[2], item[3], item[4]
		else:
			for name in self.train.columns:
				_, rmse[name], nrmse[name], mae[name], r2[name] = self._fc_score(name)
		return rmse, nrmse, mae, r2

	def ret_df(self):
		df = pd.DataFrame(columns=self.data.columns)
		df = df.append(self.best_pq, ignore_index=True).append(
			self.fc_rmse, ignore_index=True).append(self.fc_nrmse, ignore_index=True).append(
			self.fc_mae, ignore_index=True).append(self.fc_r2, ignore_index=True)
		return df.set_index(
			pd.Series(["order", 'RMSE', 'NRMSE', 'MAE', 'R2']))


if __name__ == '__main__':
	# # # solar power
	# # filepath = r'E:/HYH/code/fed_svarima/input_data/PV_data.csv'
	# # filepath = r'E:/HYH/code/fed_svarima/input_data/normalized_PVdata.csv'
	# filepath = r'input_data/power/zen_norm_PVdata.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# # data.drop(['V2', 'V3', 'V5', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V15', 'V20', 'V21', 'V23', 'V24', 'V28',
	# #            'V36', 'V39', 'V42', 'V44', 'V45'],
	# #           axis=1, inplace=True)
	# data = data.set_index(pd.to_datetime(data.index))
	# # input_data = input_data[:'2012-2-1']  # 数据过大，仅选择一年
	# # data = data.iloc[:-1, :]
	# # period = [5, 6, 7, 10, 12, 14]
	# n_test = 65  # 最后一周 #int(np.lcm.reduce(period))  # 最小公倍数; 加int防止statsmodel判断出错
	# norm = 1
	# pqs = None
	# # pqs = {'V2': (15, 0, 2), 'V3': (15, 0, 0), 'V4': (14, 0, 6), 'V5': (14, 0, 1), 'V6': (14, 0, 1), 'V7': (15, 0, 1), 'V8': (15, 0, 0), 'V9': (14, 0, 0), 'V10': (15, 0, 0), 'V11': (12, 0, 0), 'V12': (15, 0, 2), 'V13': (14, 0, 2), 'V14': (14, 0, 3), 'V15': (14, 0, 4), 'V16': (15, 0, 2), 'V17': (14, 0, 3), 'V18': (14, 0, 2), 'V19': (13, 0, 0), 'V20': (15, 0, 1), 'V21': (15, 0, 1), 'V22': (8, 0, 4), 'V23': (15, 0, 2), 'V24': (15, 0, 1), 'V25': (15, 0, 2), 'V26': (13, 0, 0), 'V27': (12, 0, 0), 'V28': (14, 0, 2), 'V29': (14, 0, 2), 'V30': (11, 0, 0), 'V31': (13, 0, 0), 'V32': (14, 0, 1), 'V33': (9, 0, 0), 'V34': (15, 0, 2), 'V35': (11, 0, 0), 'V36': (6, 0, 0), 'V37': (15, 0, 2), 'V38': (11, 0, 0), 'V39': (14, 0, 2), 'V40': (15, 0, 0), 'V41': (15, 0, 0), 'V42': (15, 0, 2), 'V43': (15, 0, 0), 'V44': (10, 0, 0), 'V45': (15, 0, 3)}

	## ghi data
	filepath = r'input_data/ghi/zen_ghi_sum_fullsites_fill0.csv'
	data = pd.read_csv(filepath, index_col=0)
	data = data.set_index(pd.to_datetime(data.index))
	n_test = 82  #  ['2017-4-30':]  # last day
	norm = 1
	pqs = None
	#
	# 保存所有输出结果，避免各种出错导致重来
	logname = os.path.basename(__file__)[:-3]
	logfile = filepath.split('/')[-1].split('.')[0]
	log_path = f'output_log/{logfile}-{logname}-n{n_test}-norm{norm}-bic.txt'
	sys.stdout = Logger(log_path, sys.stdout)
	print("log begins here")


	# grid search arma aic
	rmse_df = ARMA_opt(data, n_test, pqs=pqs).ret_df()
	rmse_df.to_csv(f'output_res/{logfile}-{logname}-n{n_test}-norm{norm}-bic.csv')
	print('arma预测结果：\n')
	print(rmse_df)
	print("log ends here")