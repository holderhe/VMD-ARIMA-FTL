# coding=gbk
import numpy as np
import pandas as pd
import psutil as ps
from joblib import Parallel
from joblib import delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from vmdpy import VMD
from output_logger import Logger
import sys, os
from util_funcs import *
from collections import Counter
import warnings

warnings.filterwarnings('error', category=RuntimeWarning,)
np.random.seed(20221021)

pd.set_option('display.max_columns', None)


from datetime import datetime
old_print = print
def timestamped_print(*args, **kwargs):
	old_print(datetime.now(), *args, **kwargs)
print = timestamped_print


class VMD_ARMA_eval:
	def __init__(self, data, n_test, pqs, clusters, K=3, method='lbfgs', norm=True, best=True, resid=True, fed=True, cpus=0.64*3.3):
		# 初始化保存所需数据
		self.cpus = cpus
		self.fed = fed
		self.resid_fc = {} if self.fed else None
		self.K = K
		self.data = data
		self.fc_size = n_test
		self.pqs = pqs
		self.clusters = clusters
		self.norm = norm
		self.resid = resid
		self.best = best
		self.method = method
		self.train, self.test = self.train_test_split()
		self.train_decomped = self.decomp_seq(self.train)
		# self.test_decomped = self.decomp_seq(self.test)
		self.train_rebuilt = self.rebuild_data(self.train_decomped)
		# self.test_rebuilt = self.rebuild_data(self.test_decomped)
		self.res_hr = {}
		self.res_default = {}

	# split a multivariate dataset into train/test sets
	def train_test_split(self):
		return self.data[:-self.fc_size], self.data[-self.fc_size:]

	def _decomp(self, data, name):
		# . some sample parameters for VMD
		alpha = 2000  # moderate bandwidth constraint
		tau = 0.  # noise-tolerance (no strict fidelity enforcement)
		# K = 10  # 3 modes
		DC = 0  # no DC part imposed
		init = 1  # initialize omegas uniformly
		tol = 1e-7

		vmd_data = data[name].dropna()
		if len(vmd_data) % 2:
			vmd_data = vmd_data[1:]
		u, u_hat, omega = VMD(vmd_data, alpha, tau, self.K, DC, init, tol)
		if not self.resid:
			return name, u
		resid = vmd_data.values-(u.sum(axis=0))
		return name, np.vstack((u, resid.reshape(1, -1)))

	def decomp_seq(self, data, parallel=True):
		data_decomped = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs)+1)
			tasks = (delayed(self._decomp)(data=data, name=name) for name in data.columns)
			res = executor(tasks)
			data_decomped = dict(res)
		else:
			for name in self.data.columns:
				data_decomped[name] = self._decomp(data, name)[1]
		if not self.resid:
			self.K -= 1
		return data_decomped

	def rebuild_data(self, data):
		rebuilt_data = {}
		for k in range(self.K+1):
			df = pd.DataFrame(columns=self.train.columns, index=self.train.index)
			for col in df.columns:
				vmd_data = self.train[col].dropna()
				if len(vmd_data) % 2:
					index1 = vmd_data.index[1:]
				else:
					index1 = vmd_data.index
				df.loc[index1, col] = data[col][k]
			rebuilt_data[k] = df
		return rebuilt_data

	def base_params_arima(self, basename, k, order, trend):
		if self.norm:
			data = RobustScaler().fit_transform(self.train_rebuilt[k][basename].dropna().values.reshape(-1, 1))
		else:
			data = self.train_decomped[basename][k].dropna()

		model = SARIMAX(data, order=order, trend=trend)
		ret_model = {}
		model_bic = {}
		for mtd in ['lbfgs', 'powell', 'nm']:
			modelhr_fitted = model.fit(maxiter=np.inf, method=mtd, maxfun=np.inf, low_memory=True)
			if not modelhr_fitted.mlefit.mle_retvals['converged']:
				continue
			params0 = np.zeros_like(modelhr_fitted.params)
			params0[-1] = 1#np.var(data)
			model0p_fitted = model.fit(start_params=params0, maxiter=np.inf, method=mtd, maxfun=np.inf, low_memory=True)
			if not model0p_fitted.mlefit.mle_retvals['converged']:
				continue
			if model0p_fitted.bic < modelhr_fitted.bic:
				ret_model[mtd] = model0p_fitted
				model_bic[mtd] = model0p_fitted.bic
			else:
				ret_model[mtd] = modelhr_fitted
				model_bic[mtd] = modelhr_fitted.bic
		try:
			model_fitted = ret_model.get(min(model_bic.items(), key=lambda x: x[1])[0])
		except:
			return None, None
		if not model_fitted.mlefit.mle_retvals['converged']:
			print(
				f'since not converged, base {basename, k} imf arima model fit for order {order, trend} with all method is skipped!\n')
			return None, None
		param = model_fitted.params
		# ret_bounds = self.set_bounds(model_fitted, model)
		print(f' {basename, k} imf arima model fit for order {order, trend} params are：{param}\n')
		return (order, trend), param#, ret_bounds

	def arima_time(self, data, scaler, name, k, order, trend, mtd):
		print(f' {name, k} imf arima model fit for order {order, trend}：\n')
		# if self.norm:
		# 	scaler = RobustScaler()
		# 	data = scaler.fit_transform(self.train_rebuilt[k][name].dropna().values.reshape(-1, 1))
		# else:
		# 	data = self.train_rebuilt[k][name].dropna()
		times, bics, iters, feval = [], [], [], []

		starttime = time.time()
		model_ = SARIMAX(data, order=order, trend=trend)
		model_start_params = model_.start_params
		model = model_.fit(start_params=model_start_params, maxiter=np.inf, method=mtd, maxfun=np.inf,
		                          low_memory=True)
		endtime = time.time()
		fitting_time = endtime - starttime
		if not model.mlefit.mle_retvals['converged']:
			if not model.mlefit.mle_retvals['converged']:
				print(
					f'{name, k} noparam fit for order {order, trend} with method {mtd} is not converged.\n')
			return None

		try:
			iters.append(model.mlefit.mle_retvals['iterations'])
		except:
			iters.append(np.nan)
		feval.append(model.mlefit.mle_retvals['fcalls'])
		bics.append(np.around(model.bic, 4))
		times.append(fitting_time)

		print(f' {name, k} imf arima model fit for order {order, trend} output bics: {bics}; time cost：{times}\n')
		if self.norm:
			model_fc = scaler.inverse_transform(model.forecast(steps=self.fc_size).reshape(-1, 1))
		else:
			model_fc = model.forecast(steps=self.fc_size)
		return np.mean(times), np.mean(bics), np.mean(iters), np.mean(feval), model_fc

	def arima_0param_time(self, data, scaler, name, k, order, trend, params, mtd):
		print(f' {name, k} imf arima model fit with 0params for order {order, trend}：\n')
		# if self.norm:
		# 	scaler = RobustScaler()
		# 	data = scaler.fit_transform(self.train_rebuilt[k][name].dropna().values.reshape(-1, 1))
		# else:
		# 	data = self.train_rebuilt[k][name].dropna()
		params0 = np.zeros_like(params)
		params0[-1] = 1#np.var(data)
		times, bics, iters, feval = [], [], [], []
		starttime = time.time()
		model = SARIMAX(data, order=order, trend=trend).fit(start_params=params0, maxiter=np.inf, method=mtd, maxfun=np.inf, low_memory=True)
		endtime = time.time()
		fitting_time = endtime - starttime
		if not model.mlefit.mle_retvals['converged']:
			if not model.mlefit.mle_retvals['converged']:
				print(
					f'{name, k} imf 0param fit for order {order, trend} with method {mtd} is not converged.\n')
			return None

		try:
			iters.append(model.mlefit.mle_retvals['iterations'])
		except:
			iters.append(np.nan)
		feval.append(model.mlefit.mle_retvals['fcalls'])
		bics.append(np.around(model.bic, 4))
		times.append(fitting_time)
		print(f' {name, k} imf arima model fit with 0params for order {order, trend} output bics: {bics}; time cost：{times}\n')
		if self.norm:
			model_fc = scaler.inverse_transform(model.forecast(steps=self.fc_size).reshape(-1, 1))
		else:
			model_fc = model.forecast(steps=self.fc_size)
		return np.mean(times), np.mean(bics), np.mean(iters), np.mean(feval), model_fc

	def arima_param_time(self, data, scaler, name, k, order, trend, param, mtd):
		print(f' {name, k} imf arima model fit with params for order {order, trend}：\n')
		# if self.norm:
		# 	scaler = RobustScaler()
		# 	data = scaler.fit_transform(self.train_rebuilt[k][name].dropna().values.reshape(-1, 1))
		# else:
		# 	data = self.train_rebuilt[k][name].dropna()
		times, bics, iters, feval = [], [], [], []

		try:
			starttime = time.time()
			model = SARIMAX(data, order=order, trend=trend).fit(start_params=param, maxiter=np.inf, method=mtd, maxfun=np.inf, low_memory=True)
			endtime = time.time()
			fitting_time = endtime - starttime
		except RuntimeWarning:
			print(
				f'since there is Exception in RuntimeWarning, {name, k} imf arima model fit for order {order, trend} with method {mtd} is skipped!\n')
			return None
		if not model.mlefit.mle_retvals['converged']:
			print(
				f'{name, k} imf param fit for order {order, trend} with method {mtd} is not converged.\n')
			return None

		try:
			iters.append(model.mlefit.mle_retvals['iterations'])
		except:
			return None
		feval.append(model.mlefit.mle_retvals['fcalls'])
		bics.append(np.around(model.bic, 4))
		times.append(fitting_time)
		print(f' {name, k} imf arima model fit with params for order {order, trend} output bics: {bics}; time cost：{times}\n')
		if self.norm:
			model_fc = scaler.inverse_transform(model.forecast(steps=self.fc_size).reshape(-1, 1))
		else:
			model_fc = model.forecast(steps=self.fc_size)
		return np.mean(times), np.mean(bics), np.mean(iters), np.mean(feval), model_fc

	def ret_time_arima_base(self, index, name, k, order, trend, params, method, mtd):
		if self.norm:
			scaler = RobustScaler()
			data = scaler.fit_transform(self.train_rebuilt[k][name].dropna().values.reshape(-1, 1))
		else:
			data = self.train_rebuilt[k][name].dropna()
			scaler = None
		if method == self.arima_time:
			return index, method(data, scaler, name, k, order, trend, mtd)
		return index, method(data, scaler, name, k, order, trend, params, mtd)
		#
		# if method == self.arima_time:
		# 	times, bics, iters, feval, model_fc = method(data, scaler, name, k, order, trend)
		# else:
		# 	times, bics, iters, feval, model_fc = method(data, scaler, name, k, order, trend, params)
		# return index, times, bics, iters, feval, model_fc

	def ret_time_arima_seq(self, name, k, order, trend, params, mtd):
		print(f' {name, k} imf arima model fit with params for order {order, trend}：\n')
		result = {}
		index = ['HR', 'Default', 'TSP']
		methods = [self.arima_time, self.arima_0param_time, self.arima_param_time]
		# if parallel:
		# 	n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04,
		# 	             (1 - ps.cpu_percent() / 100) * 32) * self.cpus
		# 	executor = Parallel(n_jobs=int(n_jobs) + 1)
		# 	tasks = (delayed(self.ret_time_arima_base)(index[i], name, k, order, trend, params, methods[i]) for i in range(len(index)))
		# 	res = executor(tasks)
		# 	result = dict(res)
		# else:
		# 	for i in range(len(index)):
		# 		result[index[i]] = self.ret_time_arima_base(index[i], name, k, order, trend, params, methods[i])[1]
		result['TSP'] = self.ret_time_arima_base(index[-1], name, k, order, trend, params, methods[-1], mtd)[1]
		if result['TSP'] is None:
			return None, None, None
		if self.res_hr.get((name, k, order, trend, mtd), None) is None:
			self.res_hr[(name, k, order, trend, mtd)] = self.ret_time_arima_base(index[0], name, k, order, trend, params, methods[0], mtd)[1]
		result['HR'] = self.res_hr[(name, k, order, trend, mtd)]
		if result['HR'] is None:
			self.res_hr.pop((name, k, order, trend, mtd))
			return None, None, None
		if self.res_default.get((name, k, order, trend, mtd), None) is None:
			self.res_default[(name, k, order, trend, mtd)] = self.ret_time_arima_base(index[1], name, k, order, trend, params, methods[1], mtd)[1]
		result['Default'] = self.res_default[(name, k, order, trend, mtd)]
		if result['Default'] is None:
			self.res_default.pop((name, k, order, trend, mtd))
			return None, None, None
		# result['HR'] = self.ret_time_arima_base(index[0], name, k, order, trend, params, methods[0])[1]
		# result['Default'] = self.ret_time_arima_base(index[0], name, k, order, trend, params, methods[1])[1]
		res_without_param, res_with_param0,	res_with_param = result['HR'], result['Default'], result['TSP']
		# res_without_param = self.arima_time(name, k, order, trend)
		# res_with_param0 = self.arima_0param_time(name, k, order, trend, params)
		# res_with_param = self.arima_param_time(name, k, order, trend, params)
		# # res_with_param_bounds = self.arima_param_bounds_time(name, k, order, trend, params, bounds)

		ret_time_noparam = res_without_param[0] - res_with_param[0]
		ret_time_0param = res_with_param0[0] - res_with_param[0]
		# ret_time_param_bounds = res_with_param[0] - res_with_param_bounds[0]

		print(f' {name, k} imf arima model fit with params compared to no params for order {order, trend} time saved：{ret_time_noparam} secs\n')
		print(f' {name, k} imf arima model fit with params compared to 0params for order {order, trend} time saved：{ret_time_0param} secs\n')
		# print(f' {name, k} imf arima model fit with params and bounds compared to params for order {order, trend} time saved：{ret_time_param_bounds} secs\n')

		return res_without_param, res_with_param0, res_with_param#, res_with_param_bounds

	def time_same_pqs(self, k, clusters, parallel=False):
		ret_time_noparam, ret_time_0param, ret_time_param = {}, {}, {}
		bic_noparam, bic_0param, bic_param = {}, {}, {}
		iter_noparam, iter_0param, iter_param = {}, {}, {}
		feval_noparam, feval_0param, feval_param = {}, {}, {}
		fcres = {'param': {}, '0param': {}, 'noparam': {}}
		def parallel_helper(ni, nj, val):
			basename = val[ni]
			n = targetname = val[nj]
			if basename == targetname or len(self.train_rebuilt[k][basename].dropna()) < len(self.train_rebuilt[k][targetname].dropna()):
				print(f"imf{k} basename == targetname or imf{k} {basename} is shorter than imf{k} {targetname}!")
				return val[ni]+val[nj], -1
			else:
				order, trend = self.pqs[targetname][k]
				print(f'target {targetname, k} imf arima model {order, trend} choose {basename, k} as baseline.\n')
				_, base_params = self.base_params_arima(basename, k, order, trend)
				if base_params is None:
					return val[ni] + val[nj], -1
				# n = val[ni + 1] if ni + 1 < len(val) else val[0]
				for mtd in ['lbfgs', 'powell', 'nm']:
					res_without_param, res_with_param0, res_with_param = \
						self.ret_time_arima_seq(n, k, order, trend, base_params, mtd)
					if res_without_param is None or res_with_param0 is None or res_with_param is None:
						print(
							f'target {targetname, k} from source {basename, k} with method {mtd} is not converged.\n')
						continue
					else:
						break

				if res_without_param is None or res_with_param0 is None or res_with_param is None:
					print(
						f'target {targetname, k} from source {basename, k} with all method is not converged. Skipped this pair.\n')
					return val[ni] + val[nj], -1
				store_key = basename + " to " + n
				return val[ni]+val[nj], (n, store_key, (res_without_param, res_with_param0, res_with_param))

		def parallel_helper2(key, val):
			if len(val) <= 1:
				print(f'since {k} imf arima model with cluster {key} is only one {val}, skip it!\n')
				pass
			else:
				result_hp2 = {}
				if parallel:
					n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04,
					             (1 - ps.cpu_percent() / 100) * 32) * self.cpus
					executor = Parallel(n_jobs=int(n_jobs) + 1)
					tasks = (delayed(parallel_helper)(ni=ni, nj=nj, val=val) for ni in range(len(val)) for nj in range(len(val)))
					res = executor(tasks)
					result_hp2 = dict(res)
				else:
					for ni in range(len(val)):
						for nj in range(len(val)):#range(ni+1, len(val)):
							# basename = val[ni]
							# n = targetname = val[nj]
							# if basename == targetname or len(self.train_rebuilt[key][basename].dropna()) < len(
							# 		self.train_rebuilt[k][targetname].dropna()):
							# 	continue
							result_hp2[val[ni]+val[nj]] = parallel_helper(ni=ni, nj=nj, val=val)[1]
			print(f'进入imf{k}下所有对比结果重复排除程序!\n')
			for kres in list(result_hp2.keys()):
				if result_hp2[kres] == -1:
					result_hp2.pop(kres)
			print(f'结束imf{k}下所有对比结果重复排除程序!\n')
			return key, result_hp2

		print(f'进入imf{k}所有对比!\n')
		c_result = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04,
			             (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(parallel_helper2)(key, val) for key, val in clusters.items())
			res = executor(tasks)
			c_result = dict(res)
		else:
			for key, val in clusters.items():
				c_result[key] = parallel_helper2(key, val)[1]
		for kres in list(c_result.keys()):
			if not c_result[kres]:
				c_result.pop(kres)
		print(f'结束imf{k}所有对比!\n')


		# for key, val in clusters.items():
		# 	if len(val) <= 1:
		# 		print(f'since {k} imf arima model with cluster {key} is only one {val}, skip it!\n')
		# 		continue
		# 	else:
		# 		result = {}
		# 		if parallel:
		# 			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04,
		# 			             (1 - ps.cpu_percent() / 100) * 16) * 0.24
		# 			executor = Parallel(n_jobs=int(n_jobs) + 1)
		# 			tasks = (delayed(parallel_helper)(ni=ni) for ni in range(len(val)))
		# 			res = executor(tasks)
		# 			result = dict(res)
		# 		else:
		# 			for ni in range(len(val)):
		# 				result[ni] = parallel_helper(ni)[1]

		def parallel_helper3(resval):
			n, store_key, (res_without_param, res_with_param0, res_with_param) = resval

			ret_time_noparam[store_key] = res_without_param[0]
			ret_time_0param[store_key] = res_with_param0[0]
			ret_time_param[store_key] = res_with_param[0]
			# ret_time_param_bounds[store_key] = res_with_param_bounds[0]

			bic_noparam[store_key] = res_without_param[1]
			bic_0param[store_key] = res_with_param0[1]
			bic_param[store_key] = res_with_param[1]
			# bic_param_bounds[store_key] = res_with_param_bounds[1]

			iter_noparam[store_key] = res_without_param[2]
			iter_0param[store_key] = res_with_param0[2]
			iter_param[store_key] = res_with_param[2]
			# iter_param_bounds[store_key] = res_with_param_bounds[2]

			feval_noparam[store_key] = res_without_param[3]
			feval_0param[store_key] = res_with_param0[3]
			feval_param[store_key] = res_with_param[3]
			# feval_param_bounds[store_key] = res_with_param_bounds[3]

			fcres['noparam'][n] = (fcres['noparam'][n] + res_without_param[-1])/2 if fcres['noparam'].get(n, None) is not None else res_without_param[-1]
			fcres['0param'][n] = (fcres['0param'][n] + res_with_param0[-1])/2 if fcres['0param'].get(n, None) is not None else res_with_param0[-1]
			fcres['param'][n] = (fcres['param'][n] + res_with_param[-1])/2 if fcres['param'].get(n, None) is not None else res_with_param[-1]
			# fcres['param_bounds'][n] = res_with_param_bounds[-1]

		# for c_key, result in c_result.items():
		# 	for ni, resval in result.items():
		# 		n, store_key, (res_without_param, res_with_param0, res_with_param) = resval
		#
		# 		ret_time_noparam[store_key] = res_without_param[0]
		# 		ret_time_0param[store_key] = res_with_param0[0]
		# 		ret_time_param[store_key] = res_with_param[0]
		# 		# ret_time_param_bounds[store_key] = res_with_param_bounds[0]
		#
		# 		bic_noparam[store_key] = res_without_param[1]
		# 		bic_0param[store_key] = res_with_param0[1]
		# 		bic_param[store_key] = res_with_param[1]
		# 		# bic_param_bounds[store_key] = res_with_param_bounds[1]
		#
		# 		iter_noparam[store_key] = res_without_param[2]
		# 		iter_0param[store_key] = res_with_param0[2]
		# 		iter_param[store_key] = res_with_param[2]
		# 		# iter_param_bounds[store_key] = res_with_param_bounds[2]
		#
		# 		feval_noparam[store_key] = res_without_param[3]
		# 		feval_0param[store_key] = res_with_param0[3]
		# 		feval_param[store_key] = res_with_param[3]
		# 		# feval_param_bounds[store_key] = res_with_param_bounds[3]
		#
		# 		fcres['noparam'][n] = res_without_param[-1]
		# 		fcres['0param'][n] = res_with_param0[-1]
		# 		fcres['param'][n] = res_with_param[-1]
		# 		# fcres['param_bounds'][n] = res_with_param_bounds[-1]

		print(f'进入imf{k}对比结果返回程序!\n')
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04,
			             (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1, backend="threading")  # , require='sharedmem') # 线程或者共享内存才能共同写入
			tasks = (delayed(parallel_helper3)(resval) for c_key, result in c_result.items() for ni, resval in result.items())
			res = executor(tasks)
		else:
			for c_key, result in c_result.items():
				for ni, resval in result.items():
					parallel_helper3(resval)
		print(f'结束imf{k}对比结果返回程序!\n')


		return k, ((ret_time_noparam, ret_time_0param, ret_time_param, #ret_time_param_bounds,
		           bic_noparam, bic_0param, bic_param, #bic_param_bounds,
		           iter_noparam, iter_0param, iter_param, #iter_param_bounds,
		            feval_noparam, feval_0param, feval_param), fcres)

	def ret_res_arima(self, parallel=False):
		if self.fed:
			imfs_k = self.K - 1
		else:
			imfs_k = self.K
		result = {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self.time_same_pqs)(k=k, clusters=self.clusters[k]) for k in range(imfs_k+1))
			res = executor(tasks)
			result = dict(res)
		else:
			for k in range(imfs_k+1):
				result[k] = self.time_same_pqs(k=k, clusters=self.clusters[k])[1]
		print(f'all arima model fit with params time saved：{result} secs\n')
		return result

	def arima_time_resid(self, name, order, trend):
		print(f' {name} resid arima model fit for order {order, trend}：\n')
		if self.norm:
			scaler = RobustScaler()
			data = scaler.fit_transform(self.train_rebuilt[self.K][name].dropna().values.reshape(-1, 1))
		else:
			data = self.train_rebuilt[self.K][name].dropna()

		for mtd in ['lbfgs', 'powell', 'nm']:
			try:
				model = SARIMAX(data, order=order, trend=trend).fit(maxiter=np.inf, method=mtd, maxfun=np.inf, low_memory=True)
			except Exception as e:
				print(e)
				print(
					f'since there is Exception, {name} resid arima model fit for order {order, trend} with method {mtd} is skipped\n')
				continue
			if model.mlefit.mle_retvals['converged']:
				break

		if self.norm:
			model_fc = scaler.inverse_transform(model.forecast(steps=self.fc_size).reshape(-1, 1))
		else:
			model_fc = model.forecast(steps=self.fc_size)
		return model_fc

	def score(self, result, name):
		if self.fed:
			order, trend = self.pqs[name][self.K]
			if self.resid_fc.get(name, None) is None:
				self.resid_fc[name] = self.arima_time_resid(name, order, trend)
			result += self.resid_fc[name]
		try:
			rmse = np.sqrt(mean_squared_error(self.test[name], result))
			nrmse = rmse / np.mean(self.test[name])
			mae = mean_absolute_error(self.test[name], result)
			r2 = r2_score(self.test[name], result)
		except:
			print(f"forecast result {name} have different number of output (1!=420)")
		print(f'optimal score for:{name} is (rmse): {rmse} \n\n')
		return name, rmse, nrmse, mae, r2

	def auto_score(self, fcres, parallel=True):
		rmse, nrmse, mae, r2 = {}, {}, {}, {}
		if parallel:
			n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.04, (1 - ps.cpu_percent() / 100) * 32) * self.cpus
			executor = Parallel(n_jobs=int(n_jobs) + 1)
			tasks = (delayed(self.score)(result=fcres[name], name=name) for name in fcres.keys())
			res = executor(tasks)
			for item in res:
				rmse[item[0]], nrmse[item[0]], mae[item[0]], r2[item[0]] = item[1], item[2], item[3], item[4]
		else:
			for name in fcres.keys():
				_, rmse[name], nrmse[name], mae[name], r2[name] = self.score(result=fcres[name], name=name)
		return rmse, nrmse, mae, r2

	def ret_fcdf(self, fcres):
		rmse, nrmse, mae, r2 = self.auto_score(fcres)
		try:
			df_fcres = pd.DataFrame(columns=sorted(fcres.keys(), key=lambda x: int(x[1:])))
		except:
			df_fcres = pd.DataFrame(columns=fcres.keys())
		df_fcres = df_fcres.append(
			rmse, ignore_index=True).append(nrmse, ignore_index=True).append(
			mae, ignore_index=True).append(r2, ignore_index=True)

		return df_fcres.set_index(
			pd.Series(['RMSE', 'NRMSE', 'MAE', 'R2']))

	def ret_df(self):
		res = self.ret_res_arima()
		fcres = {'param': {}, '0param': {}, 'noparam': {}}
		ret_df = {}
		ret_fcdf = {}
		for key, val in res.items():
			ret_df[key] = pd.DataFrame(val[0], index=[
				'time noparam', 'time 0param', 'time param', #'time param and bounds',
				'bic noparam', 'bic 0param', 'bic param', #'bic param and bounds',
				'iter noparam', 'iter 0param', 'iter param', #'iter param and bounds',
				'feval noparam', 'feval 0param', 'feval param']).T
			print(f'imf {key} arima model fit with params time saved：{ret_df[key]} secs\n')

			# if not self.resid and key == 9:
			# 	continue
			for kk,vv in val[-1].items():
				for col,vv2 in vv.items():
					fcres[kk][col] = fcres[kk].get(col, 0) + vv2
		for ke,va in fcres.items():
			ret_fcdf[ke] = self.ret_fcdf(va)
			print(f'{ke} arima model forecasting results are：{ret_fcdf[ke]} secs\n')
		return ret_df, ret_fcdf


if __name__ == '__main__':
	# wait1, wait2 = True, True
	# while wait1 or wait2:
	# 	print('waiting for cpus!')
	# 	if 1 - ps.cpu_percent() / 100 > 0.325:
	# 		wait1 = False
	# 		time.sleep(30)
	# 	else:
	# 		wait1 = True
	# 		time.sleep(300)
	# 	if 1 - ps.cpu_percent() / 100 > 0.325:
	# 		wait2 = False
	# 	else:
	# 		wait2 = True
	# 		time.sleep(30)

	# #solar power
	# filepath = r'E:/HYH/code/fed_svarima/input_data/PV_data.csv'
	# filepath = r'E:/HYH/code/fed_svarima/input_data/normalized_PVdata.csv'
	filepath = r'input_data/power/zen_norm_PVdata.csv'
	data = pd.read_csv(filepath, index_col=0)
	# data.drop(['V2', 'V36'],
	#       axis=1, inplace=True)
	data = data[
		# ['V5', 'V6', 'V12', 'V13', 'V17', 'V18', 'V19', 'V21', 'V24', 'V29', 'V41', 'V43', 'V44']
		# ['V3', 'V14', 'V16', 'V22', 'V23', 'V32', 'V34', 'V39', 'V42', 'V45']
		# ['V20', 'V26', 'V27', 'V28']
		# ['V33', 'V35', 'V37', 'V38']
		# ['V8', 'V10', 'V11', 'V25', 'V40']
		['V4', 'V30', 'V31']
		]
	data = data.set_index(pd.to_datetime(data.index))
	# input_data = input_data[:'2012-2-1']  # 数据过大，仅选择一年
	# data = data.iloc[:-1, :]
	# data = data*1e4
	# period = [5, 6, 7, 10, 12, 14]
	n_test = len(data["2013-03-06":])
	k = 10
	clusters_norm_plus = {
		0: {
			0: data.columns.to_list()},
		1: {
			0: data.columns.to_list()
		},
		2: {
			0: data.columns.to_list()},
		3: {
			0: data.columns.to_list()},
		4: {
			0: data.columns.to_list()},
		5: {
			0: data.columns.to_list()
		},
		6: {
			0: data.columns.to_list()},
		7: {
			0: data.columns.to_list()
		},
		8: {
			0: data.columns.to_list()},
		9: {
			0: data.columns.to_list()
		},
		10: {
			0: data.columns.to_list()
		}
	}
	pqs = {
		'V2': {0: ((2, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	        5: ((1, 0, 2), 'n'), 6: ((0, 0, 4), 'n'), 7: ((1, 0, 3), 'c'), 8: ((1, 0, 2), 'c'), 9: ((4, 0, 2), 'c'),
	        10: ((16, 0, 0), 'c')},
	 'V3': {0: ((4, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	        5: ((1, 0, 2), 'n'), 6: ((0, 0, 3), 'c'), 7: ((1, 0, 1), 'c'), 8: ((2, 0, 3), 'c'), 9: ((2, 0, 5), 'c'),
	        10: ((6, 0, 2), 'c')},
	 'V4': {0: ((3, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'),
	        5: ((1, 0, 2), 'n'), 6: ((0, 0, 4), 'n'), 7: ((1, 0, 1), 'c'), 8: ((5, 0, 1), 'c'), 9: ((2, 0, 0), 'c'),
	        10: ((6, 0, 2), 'c')},
	 'V5': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	        5: ((1, 0, 1), 'c'), 6: ((1, 0, 3), 'c'), 7: ((1, 0, 2), 'n'), 8: ((1, 0, 1), 'n'), 9: ((7, 0, 1), 'c'),
	        10: ((17, 0, 1), 'c')},
	 'V6': {0: ((2, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	        5: ((1, 0, 1), 'c'), 6: ((1, 0, 3), 'c'), 7: ((4, 0, 1), 'c'), 8: ((2, 0, 4), 'c'), 9: ((9, 0, 1), 'c'),
	        10: ((3, 0, 0), 'n')},
	 'V7': {0: ((3, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	        5: ((1, 0, 2), 'n'), 6: ((1, 0, 1), 'n'), 7: ((5, 0, 1), 'c'), 8: ((2, 0, 4), 'n'), 9: ((2, 0, 0), 'c'),
	        10: ((7, 0, 1), 'c')},
	 'V8': {0: ((4, 0, 1), 'c'), 1: ((0, 0, 3), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'),
	        5: ((1, 0, 2), 'n'), 6: ((1, 0, 1), 'c'), 7: ((5, 0, 1), 'c'), 8: ((2, 0, 3), 'c'), 9: ((6, 0, 1), 'c'),
	        10: ((1, 0, 0), 'n')},
	 'V9': {0: ((3, 0, 2), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	        5: ((1, 0, 1), 'c'), 6: ((0, 0, 4), 'n'), 7: ((2, 0, 3), 'c'), 8: ((2, 0, 4), 'c'), 9: ((3, 0, 0), 'c'),
	        10: ((24, 0, 0), 'c')},
	 'V10': {0: ((2, 0, 4), 'n'), 1: ((1, 0, 1), 'n'), 2: ((0, 0, 3), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'n'), 7: ((1, 0, 1), 'c'), 8: ((5, 0, 1), 'c'), 9: ((2, 0, 0), 'c'),
	         10: ((6, 0, 0), 'n')},
	 'V11': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((0, 0, 3), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 3), 'n'),
	         10: ((0, 0, 1), 'n')},
	 'V12': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((0, 0, 3), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'n'), 7: ((2, 0, 1), 'c'), 8: ((5, 0, 1), 'c'), 9: ((2, 0, 0), 'c'),
	         10: ((17, 0, 0), 'c')},
	 'V13': {0: ((2, 0, 2), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 3), 'n'), 6: ((1, 0, 1), 'n'), 7: ((1, 0, 1), 'c'), 8: ((2, 0, 4), 'c'), 9: ((4, 0, 0), 'c'),
	         10: ((9, 0, 0), 'c')},
	 'V14': {0: ((2, 0, 3), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 2), 'n'), 6: ((0, 0, 3), 'n'), 7: ((2, 0, 1), 'n'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 3), 'c'),
	         10: ((21, 0, 0), 'c')},
	 'V15': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'c'), 9: ((1, 0, 0), 'c'),
	         10: ((1, 0, 0), 'n')},
	 'V16': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 2), 'n'), 6: ((0, 0, 3), 'n'), 7: ((1, 0, 2), 'n'), 8: ((1, 0, 1), 'n'), 9: ((9, 0, 0), 'c'),
	         10: ((27, 0, 0), 'c')},
	 'V17': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'n'), 7: ((2, 0, 3), 'c'), 8: ((2, 0, 4), 'c'), 9: ((1, 0, 0), 'c'),
	         10: ((10, 0, 0), 'c')},
	 'V18': {0: ((2, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 3), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 2), 'n'), 8: ((1, 0, 1), 'c'), 9: ((5, 0, 1), 'c'),
	         10: ((25, 0, 2), 'c')},
	 'V19': {0: ((2, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'n'), 6: ((1, 0, 3), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((9, 0, 1), 'c'),
	         10: ((17, 0, 0), 'c')},
	 'V20': {0: ((2, 0, 3), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'c'), 6: ((0, 0, 4), 'n'), 7: ((2, 0, 4), 'c'), 8: ((2, 0, 4), 'n'), 9: ((3, 0, 3), 'c'),
	         10: ((13, 0, 1), 'c')},
	 'V21': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'n'), 7: ((3, 0, 3), 'c'), 8: ((2, 0, 5), 'c'), 9: ((3, 0, 0), 'c'),
	         10: ((15, 0, 3), 'c')},
	 'V22': {0: ((3, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((0, 0, 3), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'c'), 6: ((0, 0, 4), 'n'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 3), 'c'),
	         10: ((17, 0, 3), 'c')},
	 'V23': {0: ((5, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((0, 0, 4), 'n'), 6: ((0, 0, 4), 'n'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 3), 'c'),
	         10: ((20, 0, 2), 'c')},
	 'V24': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((2, 0, 2), 'n'), 6: ((0, 0, 3), 'n'), 7: ((1, 0, 1), 'c'), 8: ((2, 0, 4), 'n'), 9: ((12, 0, 1), 'c'),
	         10: ((24, 0, 0), 'c')},
	 'V25': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'n'), 7: ((1, 0, 1), 'c'), 8: ((2, 0, 4), 'n'), 9: ((2, 0, 2), 'n'),
	         10: ((3, 0, 0), 'c')},
	 'V26': {0: ((2, 0, 4), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'n'), 7: ((2, 0, 1), 'c'), 8: ((3, 0, 3), 'c'), 9: ((5, 0, 1), 'c'),
	         10: ((4, 0, 2), 'c')},
	 'V27': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 4), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'),
	         10: ((15, 0, 0), 'n')},
	 'V28': {0: ((3, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((1, 0, 2), 'n'), 8: ((2, 0, 4), 'c'), 9: ((2, 0, 6), 'n'),
	         10: ((1, 0, 0), 'n')},
	 'V29': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 3), 'c'), 7: ((1, 0, 1), 'c'), 8: ((2, 0, 4), 'c'), 9: ((2, 0, 6), 'c'),
	         10: ((1, 0, 0), 'n')},
	 'V30': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'c'), 9: ((3, 0, 2), 'c'),
	         10: ((3, 0, 0), 'n')},
	 'V31': {0: ((2, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((0, 0, 3), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'n'), 6: ((0, 0, 3), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 1), 'c'),
	         10: ((3, 0, 0), 'n')},
	 'V32': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'n'), 6: ((0, 0, 5), 'n'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 3), 'c'),
	         10: ((1, 0, 0), 'n')},
	 'V33': {0: ((2, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((0, 0, 3), 'c'), 7: ((0, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((6, 0, 1), 'c'),
	         10: ((25, 0, 0), 'c')},
	 'V34': {0: ((2, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 2), 'n'), 6: ((0, 0, 3), 'n'), 7: ((1, 0, 2), 'n'), 8: ((1, 0, 1), 'c'), 9: ((2, 0, 3), 'c'),
	         10: ((16, 0, 0), 'c')},
	 'V35': {0: ((2, 0, 2), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 1), 'c'),
	         10: ((3, 0, 0), 'n')},
	 'V36': {0: ((1, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'n'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 2), 'c'),
	         10: ((6, 0, 3), 'c')},
	 'V37': {0: ((3, 0, 2), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((2, 0, 3), 'c'), 8: ((1, 0, 1), 'c'), 9: ((2, 0, 5), 'c'),
	         10: ((3, 0, 0), 'n')},
	 'V38': {0: ((2, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((0, 0, 4), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((5, 0, 0), 'c'),
	         10: ((9, 0, 0), 'c')},
	 'V39': {0: ((3, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((5, 0, 2), 'n'), 6: ((0, 0, 4), 'n'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((4, 0, 1), 'n'),
	         10: ((16, 0, 0), 'c')},
	 'V40': {0: ((5, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((0, 0, 2), 'n'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'n'), 7: ((5, 0, 1), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 0), 'c'),
	         10: ((5, 0, 3), 'c')},
	 'V41': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'n'), 6: ((0, 0, 3), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'c'), 9: ((7, 0, 1), 'c'),
	         10: ((3, 0, 0), 'c')},
	 'V42': {0: ((2, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 2), 'n'), 8: ((1, 0, 1), 'n'), 9: ((10, 0, 1), 'c'),
	         10: ((3, 0, 0), 'n')},
	 'V43': {0: ((2, 0, 1), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 3), 'n'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 1), 'c'),
	         5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'c'), 9: ((7, 0, 1), 'c'),
	         10: ((3, 0, 0), 'n')},
	 'V44': {0: ((4, 0, 0), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 1), 'c'), 4: ((1, 0, 2), 'n'),
	         5: ((4, 0, 2), 'c'), 6: ((0, 0, 4), 'n'), 7: ((0, 0, 3), 'c'), 8: ((1, 0, 1), 'n'), 9: ((2, 0, 3), 'n'),
	         10: ((4, 0, 0), 'c')},
	 'V45': {0: ((2, 0, 2), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'c'), 3: ((1, 0, 2), 'n'), 4: ((1, 0, 2), 'n'),
	         5: ((1, 0, 1), 'n'), 6: ((0, 0, 3), 'c'), 7: ((4, 0, 1), 'c'), 8: ((2, 0, 4), 'c'), 9: ((2, 0, 4), 'c'),
	         10: ((10, 0, 1), 'c')}
	}

	# # # ghi
	# filepath = r'input_data/ghi/zen_ghi_sum_fullsites_fill0.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# data = data.set_index(pd.to_datetime(data.index))
	# n_test = len(data['2017-4-30':])  # last day
	# k = 14  # to split up period central frequency
	# # pqs = None
	# clusters_norm_plus = {
	# 	0: {
	# 		0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	1: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	2: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	3: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	4: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	5: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	6: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	7: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	8: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur', 'Lahore']},
	# 	9: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur',
	# 	        'Lahore']},
	# 	10: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur',
	# 	        'Lahore']},
	# 	11: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur',
	# 	        'Lahore']},
	# 	12: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur',
	# 	        'Lahore']},
	# 	13: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur',
	# 	        'Lahore']},
	# 	14: {0: ['Multan', 'Peshawar', 'Hyderabad', 'Karachi', 'Quetta', 'Khuzdar', 'Islamabad', 'Bahawalpur',
	# 	        'Lahore']},
	# }
	# pqs = {
	# 	'Peshawar': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'),
	#               4: ((1, 0, 1), 'n'), 5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'),
	#               8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'), 10: ((1, 0, 1), 'n'), 11: ((1, 0, 1), 'c'),
	#               12: ((1, 0, 1), 'c'), 13: ((22, 0, 0), 'c'), 14: ((2, 0, 4), 'n')},
	#  'Islamabad': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'),
	#                4: ((1, 0, 1), 'n'), 5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'),
	#                8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'), 10: ((1, 0, 1), 'n'), 11: ((1, 0, 1), 'n'),
	#                12: ((2, 0, 2), 'c'), 13: ((3, 0, 0), 'c'), 14: ((3, 0, 1), 'n')},
	#  'Lahore': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'), 4: ((1, 0, 1), 'n'),
	#             5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((0, 0, 2), 'n'), 8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'),
	#             10: ((2, 0, 4), 'c'), 11: ((1, 0, 1), 'c'), 12: ((3, 0, 2), 'c'), 13: ((4, 0, 0), 'c'),
	#             14: ((2, 0, 4), 'c')},
	#  'Quetta': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'), 4: ((1, 0, 1), 'n'),
	#             5: ((1, 0, 1), 'c'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'), 8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'),
	#             10: ((1, 0, 1), 'n'), 11: ((1, 0, 1), 'n'), 12: ((2, 0, 3), 'c'), 13: ((2, 0, 1), 'c'),
	#             14: ((3, 0, 2), 'c')},
	#  'Multan': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'), 4: ((1, 0, 1), 'n'),
	#             5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((0, 0, 2), 'n'), 8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'),
	#             10: ((1, 0, 1), 'n'), 11: ((1, 0, 1), 'c'), 12: ((4, 0, 1), 'c'), 13: ((2, 0, 0), 'c'),
	#             14: ((2, 0, 1), 'c')},
	#  'Bahawalpur': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'),
	#                 4: ((1, 0, 1), 'n'), 5: ((1, 0, 1), 'n'), 6: ((0, 0, 3), 'n'), 7: ((1, 0, 1), 'c'),
	#                 8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'), 10: ((1, 0, 1), 'n'), 11: ((5, 0, 1), 'c'),
	#                 12: ((5, 0, 1), 'c'), 13: ((4, 0, 2), 'n'), 14: ((7, 0, 1), 'c')},
	#  'Khuzdar': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'),
	#              4: ((1, 0, 1), 'n'), 5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'),
	#              8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'), 10: ((1, 0, 1), 'n'), 11: ((1, 0, 1), 'n'),
	#              12: ((6, 0, 1), 'c'), 13: ((2, 0, 0), 'c'), 14: ((6, 0, 1), 'c')},
	#  'Hyderabad': {0: ((1, 0, 1), 'c'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'),
	#                4: ((1, 0, 1), 'n'), 5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'),
	#                8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'), 10: ((1, 0, 1), 'n'), 11: ((1, 0, 1), 'n'),
	#                12: ((5, 0, 1), 'c'), 13: ((5, 0, 1), 'c'), 14: ((2, 0, 1), 'n')},
	#  'Karachi': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'),
	#              4: ((1, 0, 1), 'n'), 5: ((1, 0, 1), 'n'), 6: ((1, 0, 1), 'c'), 7: ((1, 0, 1), 'c'),
	#              8: ((1, 0, 1), 'c'), 9: ((1, 0, 1), 'c'), 10: ((1, 0, 1), 'n'), 11: ((1, 0, 1), 'c'),
	#              12: ((2, 0, 2), 'c'), 13: ((2, 0, 1), 'c'), 14: ((4, 0, 1), 'n')}
	# }


	#  #pv
	# filepath = r'input_data/household/zen_household_data_60min_pv_cut.csv'
	# data = pd.read_csv(filepath, index_col=0)
	# data = data.set_index(pd.to_datetime(data.index))
	# data.drop(['DE_KN_industrial2_pv'], axis=1, inplace=True)
	# n_test = len(data['2017-03-06':])  # last day
	# k = 7  # to split up period and keep stationary
	# pqs = {
	# 	'DE_KN_industrial1_pv_1': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 15), 'c'), 4: ((1, 0, 7), 'n'), 5: ((1, 0, 1), 'n'), 6: ((1, 0, 4), 'n'), 7: ((7, 0, 0), 'c')}, 'DE_KN_industrial1_pv_2': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 0), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'), 4: ((1, 0, 6), 'n'), 5: ((1, 0, 5), 'n'), 6: ((1, 0, 3), 'n'), 7: ((17, 0, 0), 'n')}, 'DE_KN_industrial3_pv_facade': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'), 4: ((1, 0, 7), 'n'), 5: ((1, 0, 5), 'c'), 6: ((1, 0, 3), 'n'), 7: ((6, 0, 0), 'c')}, 'DE_KN_industrial3_pv_roof': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 12), 'n'), 3: ((1, 0, 7), 'c'), 4: ((1, 0, 6), 'n'), 5: ((1, 0, 5), 'n'), 6: ((1, 0, 3), 'n'), 7: ((4, 0, 0), 'c')}, 'DE_KN_residential1_pv': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 5), 'n'), 4: ((1, 0, 14), 'c'), 5: ((1, 0, 5), 'c'), 6: ((1, 0, 3), 'n'), 7: ((4, 0, 0), 'c')}, 'DE_KN_residential3_pv': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'), 4: ((1, 0, 1), 'n'), 5: ((1, 0, 5), 'c'), 6: ((1, 0, 4), 'n'), 7: ((4, 0, 0), 'n')}, 'DE_KN_residential4_pv': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'n'), 2: ((1, 0, 12), 'c'), 3: ((1, 0, 14), 'c'), 4: ((1, 0, 1), 'n'), 5: ((1, 0, 5), 'n'), 6: ((1, 0, 4), 'n'), 7: ((5, 0, 0), 'c')}, 'DE_KN_residential6_pv': {0: ((1, 0, 0), 'n'), 1: ((1, 0, 1), 'c'), 2: ((1, 0, 1), 'n'), 3: ((1, 0, 1), 'n'), 4: ((1, 0, 1), 'n'), 5: ((1, 0, 6), 'n'), 6: ((1, 0, 4), 'n'), 7: ((2, 0, 1), 'c')}
	# }
	# clusters_norm_plus = {
	# 	0: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	1: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	2: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	3: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	4: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	5: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	6: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	7: {
	# 		0: ['DE_KN_residential1_pv', 'DE_KN_residential4_pv', 'DE_KN_industrial3_pv_facade', 'DE_KN_industrial3_pv_roof', 'DE_KN_industrial1_pv_1', 'DE_KN_industrial1_pv_2', 'DE_KN_residential6_pv', 'DE_KN_residential3_pv']},
	# 	}

	method = 'lbfgs'
	norm = 1
	resid = 1
	best = 0

	logname = os.path.basename(__file__)[:-3]
	logfile = filepath.split('/')[-1].split('.')[0]
	log_path = f'output_log/{logfile}-{logname}-n{n_test}-k{k}-cluster5-norm{norm}-{method}-best{best}-resid{resid}-7.txt'
	sys.stdout = Logger(log_path, sys.stdout)
	print(f"log begins here \n setting are {logfile}-{logname}-n{n_test}-k{k}-cluster5-norm{norm}-{method}-best{best}-resid{resid}")

	model= VMD_ARMA_eval(data, n_test, pqs=pqs, K=k, clusters=clusters_norm_plus, method=method, norm=norm, best=best, resid=resid)
	ret_df, rmse_df = model.ret_df()
	writer1 = pd.ExcelWriter(f'output_res/{logfile}-{logname}-n{n_test}-k{k}-cluster5-norm{norm}-{method}-best{best}-resid{resid}-7v4.xlsx')  # 重点1：writer不能在下面的for循环中
	for key, val in ret_df.items():
		val.to_excel(writer1, 'imf' + str(key))
	writer1.save()  # 重点2：save不能在for循环里
	writer1.close()

	writer2 = pd.ExcelWriter(f'output_res/{logfile}-{logname}-n{n_test}-k{k}-cluster5-norm{norm}-{method}-best{best}-resid{resid}-fcres-7v4.xlsx')  # 重点1：writer不能在下面的for循环中
	for ke, resdf in rmse_df.items():
		resdf.to_excel(writer2, ke)
	writer2.save()  # 重点2：save不能在for循环里
	writer2.close()
	print("log ends here")