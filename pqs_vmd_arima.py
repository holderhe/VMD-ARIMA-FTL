import time

import pandas as pd
import psutil as ps
from joblib import Parallel
from joblib import delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsforecast.arima import auto_arima_f
from vmdpy import VMD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer

from output_logger import Logger
import sys, os
from util_funcs import *
from collections import Counter


pd.set_option('display.max_columns', None)



class VMD_ARMA_opt:
	def __init__(self, data, n_test, K=3, model='decom', pqs=None, norm=True, diff=False, cpus=0.64*0.3):
		# 初始化保存所需数据
		self.cpus = cpus
		self.diff = diff
		self.K = K
		self.norm = norm
		self.data = data
		self.fc_size = n_test
		self.train, self.test = self.train_test_split()
		self.train_decomped = self.decomp_seq()
		if pqs is None:
			self.orders = self.auto_order()
			if model == 'fed':
				self.best_pq = self.pq_consist_fed(self.auto_pq())
			elif model == 'trans':
				self.pq_consist_trans(self.auto_pq())
			else:
				self.best_pq = self.auto_pq()
		else:
			if model == 'fed':
				self.best_pq = self.pq_consist_fed(pqs)
			elif model == 'trans':
				self.pq_consist_trans(pqs)
			else:
				self.best_pq = pqs
		self.fc_rmse, self.fc_nrmse, self.fc_mae, self.fc_r2 = self.auto_score()

	# split a multivariate dataset into train/test sets
	def train_test_split(self):
		return self.data[:-self.fc_size], self.data[-self.fc_size:]

	def pq_consist_trans(self, pqs):  # transfer means consistent parameter in all imf_k
		pqsn = {}
		cons = {}
		for k1, v1 in pqs.items():
			for a, b in v1.items():
				cons[a] = cons.get(a, []) + [b]
		for k2, v2 in pqs.items():
			for a, b in v2.items():
				pqsn[k2] = pqsn.get(k2, {})
				pqsn[k2][a] = sorted(Counter(cons[a]).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]
		return pqsn

	def pq_consist_fed(self, pqs):  # transfer means consistent parameter in all imf_k
		pqsn = {}
		cons = {}
		for k1, v1 in pqs.items():
			for a, b in v1.items():
				cons[a] = cons.get(a, []) + [b]
		for k2, v2 in pqs.items():
			for a, b in v2.items():
				pqsn[k2] = pqsn.get(k2, {})
				if a == 0 or a == self.K:  # federation left out first imf and residual seq
					pqsn[k2][a] = pqs[k2][a]
				else:
					pqsn[k2][a] = sorted(Counter(cons[a]).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]
		return pqsn

	def _decomp(self, name):
		# . some sample parameters for VMD
		alpha = 2000  # moderate bandwidth constraint
		tau = 0.  # noise-tolerance (no strict fidelity enforcement)
		# K = 10  # 3 modes
		DC = 0  # no DC part imposed
		init = 1  # initialize omegas uniformly
		tol = 1e-7
		vmd_data = self.train[name].dropna().values
		if len(vmd_data) % 2:
			vmd_data = vmd_data[1:]
		u, u_hat, omega = VMD(vmd_data, alpha, tau, self.K, DC, init, tol)
		resid = vmd_data-(u.sum(axis=0))
		return name, np.vstack((u, resid.reshape(1, -1)))

	def decomp_seq(self, parallel=True):
		data_decomped = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs)+1)
			tasks = (delayed(self._decomp)(name=name) for name in self.train.columns)
			res = executor(tasks)
			data_decomped = dict(res)
		else:
			for name in self.data.columns:
				data_decomped[name] = self._decomp(name)[1]
		return data_decomped

	def get_max_order_imf(self, k):
		orders = []
		for name in self.data.columns:
			print(f'get period for : {name, k}')
			orders += set(get_period_welch(self.train_decomped[name][k]))
		orders = sorted(Counter(orders).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		if not orders:
			orders_global = []
			for name in self.data.columns:
				print(f'get period for {name} since all imf in {name} are []!')
				orders_global += set(get_period_welch(self.train[name].dropna()))
			orders_global = sorted(Counter(orders_global).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
			max_order = orders_global[0][0]
		else:
			max_order = orders[0][0]
		return k, max_order

	def auto_order(self, parallel=True):
		orders = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self.get_max_order_imf)(k=k) for k in range(self.K+1))
			res = executor(tasks)
			orders = dict(res)
		else:
			for name in self.train.columns:
				orders[name] = self.get_max_order_imf(name)[1]
		return orders

	def _pq_imf1(self, name, k):
		print(f'Searching order of p and q for : {name, k}')
		if self.diff and k == 0:
			diff = 1
		else:
			diff = 0
		stepwise_model = auto_arima(self.train_decomped[name][k], start_p=0, start_q=0,
		                            max_p=self.orders[k]+100, max_q=self.orders[k]+100, d=diff, seasonal=False,
		                            trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
		                            information_criterion ='bic',maxiter=1000)
		parameter = stepwise_model.get_params().get('order')
		print(f'optimal order for:{name,k} is: {parameter} \n\n')
		return k, parameter

	def _pq_imf(self, name, k):
		print(f'Searching order of p and q for : {name, k}')
		if self.norm:
			scaler = RobustScaler()
			data = scaler.fit_transform(self.train_decomped[name][k].reshape(-1, 1)).flatten()
		else:
			data = self.train_decomped[name][k]
		if self.diff and k == 0:
			diff = 1
		else:
			diff = 0
		stepwise_model = auto_arima_f(x=data, start_p=0, start_q=0,
		                            max_p=self.orders[k]+100, max_q=self.orders[k]+100, d=diff, seasonal=False,
		                            num_cores=4,trace=True, ic='bic')
		p, q, P, Q, s, d, D = stepwise_model.get('arma')
		order = (p, d, q)
		seasonal_order = (P, D, Q, s)
		if stepwise_model.get('coef').get('intercept', None) is not None:
			trend = 'c'
		else: trend = 'n'
		print(f'optimal order for:{name, k} is: {order} with trend={trend} \n\n')
		return name, (order, trend)

	def _pq_seq(self, name, parallel=False):
		pq_seq = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self._pq_imf)(name=name, k=k) for k in range(self.K+1))
			res = executor(tasks)
			pq_seq = dict(res)
		else:
			for k in range(self.K+1):
				pq_seq[k] = self._pq_imf(name, k)[1]
		return name, pq_seq

	def auto_pq(self, parallel=True):
		pq = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self._pq_seq)(name=name) for name in self.train.columns)
			res = executor(tasks)
			pq = dict(res)
		else:
			for name in self.train.columns:
				pq[name] = self._pq_seq(name)[1]
		print(f'optimal all data orders for vmd-arima is: {pq} \n\n')
		return pq

	def _fc_imf(self, name, k):
		order, trend = self.best_pq[name][k]
		if self.norm:
			scaler = RobustScaler()
			data = scaler.fit_transform(self.train_decomped[name][k].reshape(-1, 1))
		else:
			data = self.train_decomped[name][k]
		model = SARIMAX(data, order=order, trend=trend).fit(maxiter=np.inf, method='powell', maxfun=np.inf)
		if self.norm:
			result = scaler.inverse_transform(model.forecast(steps=self.fc_size).reshape(-1, 1))
		else:
			result = model.forecast(steps=self.fc_size)
		return result

	def _fc_score(self, name, parallel=True):
		result = 0#np.zeros(self.fc_size)
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self._fc_imf)(name=name, k=k) for k in range(self.K+1))
			res = executor(tasks)
			result = np.sum(res, axis=0)
		else:
			for k in range(self.K+1):
				result += self._fc_imf(name, k)
		rmse = np.sqrt(mean_squared_error(self.test[name], result))
		nrmse = rmse / np.mean(self.test[name])
		mae = mean_absolute_error(self.test[name], result)
		r2 = r2_score(self.test[name], result)
		print(f'optimal score for:{name} is (rmse): {rmse} \n\n')
		return name, rmse, nrmse, mae, r2

	def auto_score(self, parallel=True):
		rmse, nrmse, mae, r2 = {}, {}, {}, {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
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
		df = df.append(
			self.fc_rmse, ignore_index=True).append(self.fc_nrmse, ignore_index=True).append(
			self.fc_mae, ignore_index=True).append(self.fc_r2, ignore_index=True)
		return df.set_index(
			pd.Series(['RMSE', 'NRMSE', 'MAE', 'R2']))


if __name__ == '__main__':
	model_list = ['decom', 'trans', 'fed']
	# # # # solar power
	# # filepath = r'E:/HYH/code/fed_svarima/input_data/PV_data.csv'
	# # filepath = r'E:/HYH/code/fed_svarima/input_data/normalized_PVdata.csv'
	# filepath = r'input_data/power/zen_norm_PVdata.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# # data.drop(['V2', 'V9', 'V16', 'V27', 'V36'],
	# #           axis=1, inplace=True)
	# data = data.set_index(pd.to_datetime(data.index))
	# # input_data = input_data[:'2012-2-1']  # 数据过大，仅选择一年
	# # data = data.iloc[:-1, :]
	# # data = data*1e4
	#
	# # period = [5, 6, 7, 10, 12, 14]
	# n_test = len(data["2013-03-06":])  # 最后一周 int(np.lcm.reduce(period))  # 最小公倍数; 加int防止statsmodel判断出错
	# k = 10  # to split up period
	# # k = 4  # only for welch+solar
	# model = model_list[0]
	# norm = 1
	# diff = False
	# pqs = None
	# # # pqs = {'V2': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((3, 0, 3), 'c')}, 'V3': {0: ((1, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 4), 'c'), 7: ((6, 0, 0), 'n'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 0), 'c')}, 'V4': {0: ((7, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 3), 'n'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((8, 0, 0), 'c')}, 'V5': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 6), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((6, 0, 1), 'c')}, 'V6': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 6), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((21, 0, 2), 'c')}, 'V7': {0: ((5, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((3, 0, 4), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'c')}, 'V8': {0: ((6, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((6, 0, 1), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((0, 0, 0), 'n')}, 'V9': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((8, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((10, 0, 1), 'c')}, 'V10': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 4), 'c'), 7: ((2, 0, 2), 'c'), 8: ((1, 0, 1), 'n'), 9: ((6, 0, 1), 'n')}, 'V11': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((4, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((1, 0, 3), 'n')}, 'V12': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((7, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 1), 'c')}, 'V13': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((2, 0, 4), 'c'), 8: ((4, 0, 1), 'c'), 9: ((1, 0, 0), 'n')}, 'V14': {0: ((3, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((7, 0, 1), 'n'), 8: ((1, 0, 1), 'n'), 9: ((22, 0, 0), 'c')}, 'V15': {0: ((4, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 4), 'n'), 7: ((2, 0, 2), 'n'), 8: ((1, 0, 1), 'n'), 9: ((4, 0, 3), 'c')}, 'V16': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 6), 'c'), 7: ((3, 0, 4), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'n')}, 'V17': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'n'), 6: ((2, 0, 4), 'c'), 7: ((10, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((21, 0, 2), 'c')}, 'V18': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((12, 0, 0), 'c'), 8: ((1, 0, 1), 'n'), 9: ((9, 0, 1), 'c')}, 'V19': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 2), 'n'), 3: ((1, 0, 2), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((5, 0, 1), 'c')}, 'V20': {0: ((2, 0, 2), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 3), 'n'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((26, 0, 0), 'c')}, 'V21': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((3, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((21, 0, 0), 'c')}, 'V22': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((5, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((25, 0, 1), 'n')}, 'V23': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((5, 0, 0), 'c'), 8: ((1, 0, 1), 'n'), 9: ((26, 0, 1), 'c')}, 'V24': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((8, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'n')}, 'V25': {0: ((5, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((2, 0, 2), 'n'), 8: ((1, 0, 1), 'n'), 9: ((0, 0, 0), 'n')}, 'V26': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 3), 'c'), 7: ((2, 0, 3), 'c'), 8: ((2, 0, 1), 'c'), 9: ((0, 0, 0), 'n')}, 'V27': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 4), 'c'), 7: ((1, 0, 1), 'c'), 8: ((2, 0, 1), 'c'), 9: ((9, 0, 1), 'c')}, 'V28': {0: ((3, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((3, 0, 4), 'c'), 7: ((6, 0, 0), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'c')}, 'V29': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 3), 'c'), 6: ((3, 0, 4), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'n')}, 'V30': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 2), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((1, 0, 0), 'n')}, 'V31': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((3, 0, 3), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((6, 0, 0), 'c')}, 'V32': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 6), 'c'), 7: ((12, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'c')}, 'V33': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'c')}, 'V34': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((5, 0, 0), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 0), 'c')}, 'V35': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 4), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'c')}, 'V36': {0: ((1, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'n'), 7: ((1, 0, 1), 'n'), 8: ((2, 0, 0), 'c'), 9: ((11, 0, 1), 'c')}, 'V37': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 2), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 2), 'c')}, 'V38': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 2), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 6), 'c'), 7: ((11, 0, 0), 'c'), 8: ((1, 0, 1), 'n'), 9: ((15, 0, 1), 'c')}, 'V39': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 2), 'c'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((12, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((21, 0, 1), 'c')}, 'V40': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 4), 'n'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((0, 0, 0), 'n')}, 'V41': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 2), 'c'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((2, 0, 3), 'n'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((1, 0, 0), 'n')}, 'V42': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 6), 'c'), 7: ((6, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((9, 0, 1), 'c')}, 'V43': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((7, 0, 1), 'c')}, 'V44': {0: ((4, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((2, 0, 6), 'c'), 8: ((3, 0, 0), 'c'), 9: ((6, 0, 0), 'c')}, 'V45': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 2), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'), 5: ((1, 0, 1), 'c'), 6: ((0, 0, 6), 'c'), 7: ((2, 0, 6), 'c'), 8: ((1, 0, 1), 'n'), 9: ((20, 0, 1), 'c')}}

	# # ghi data
	# filepath = r'input_data/ghi/zen_ghi_sum_fullsites_fill0.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# data = data.set_index(pd.to_datetime(data.index))
	# n_test = len(data['2017-4-30':])  # last day
	# k = 14  # to split up period central frequency
	# model = model_list[0]
	# norm = 1
	# diff = False
	# pqs = None


	#washing_machine
	filepath = r'input_data/household/household_data_60min_washing_machine_cut.csv'
	data = pd.read_csv(filepath, index_col=0)
	data = data.set_index(pd.to_datetime(data.index))
	n_test = len(data['2017-03-07':])  # last day
	k = 10  # to split up period and keep stationary
	model = model_list[0]
	norm = 1
	diff = True
	pqs = None

	#  #pv
	# filepath = r'input_data/household/zen_household_data_60min_pv_cut.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# data = data.set_index(pd.to_datetime(data.index))
	# data.drop(['DE_KN_industrial2_pv'], axis=1, inplace=True)
	# n_test = len(data['2017-03-06':])  # last day
	# k = 7  # to split up period and keep stationary
	# model = model_list[0]
	# norm = 1
	# diff = False
	# pqs = None

	# # #  #wind_speed
	# filepath = r'input_data/ghi/wind_speed_sum_fullsites_cut.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# data = data.set_index(pd.to_datetime(data.index))
	# n_test = len(data['2017-4-30':])  # last day
	# k = 12  # to split up period and keep stationary
	# model = model_list[0]
	# norm = 1
	# diff = False
	# pqs = None

	# # #  #grid_import
	# filepath = r'input_data/household/household_data_60min_gridimport_cut.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# data = data.set_index(pd.to_datetime(data.index))
	# n_test = len(data['2016-11-22':])  # last day
	# k = 12  # to split up period and keep stationary
	# model = model_list[0]
	# norm = 1
	# diff = True
	# pqs = None

	wait1, wait2 = True, True
	while wait1 or wait2:
		print('waiting for cpus!')
		if 1 - ps.cpu_percent() / 100 > 0.25:
			wait1 = False
			time.sleep(30)
		else:
			wait1 = True
			time.sleep(300)
		if 1 - ps.cpu_percent() / 100 > 0.25:
			wait2 = False
		else:
			wait2 = True
			time.sleep(30)



	# 保存所有输出结果，避免各种出错导致重来
	logname = os.path.basename(__file__)[:-3]
	logfile = filepath.split('/')[-1].split('.')[0]
	log_path = f'output_log/{logfile}-{logname}-m{model}-n{n_test}-k{k}-norm{norm}-bic.txt'
	sys.stdout = Logger(log_path, sys.stdout)
	print("log begins here")


	# grid search vmd-arma aic
	rmse_df = VMD_ARMA_opt(data, n_test, K=k, pqs=pqs, model=model, diff=diff).ret_df()
	rmse_df.to_csv(f'output_res/{logfile}-{logname}-m{model}-n{n_test}-k{k}-norm{norm}-bic.csv')
	print('vmd-arma预测结果：\n')
	print(rmse_df)
	print("log ends here")