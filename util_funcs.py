import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, signal, stats
from pmdarima import auto_arima
from statsforecast.arima import auto_arima_f
from collections import Counter
from vmdpy import VMD
from joblib import Parallel, delayed
import psutil as ps


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def get_period_welch(data):
	"""
	solar power imf0 not significant, loose percentile and prominence
	"""
	def linear(x, a, b):
		return a * x + b

	# def plotmax(arr, arr_x, arr_y):
	# 	for ind in arr:
	# 		show = '(' + str(round(1 / arr_x[ind], 3)) + ')'
	# 		plt.scatter(arr_x[ind], arr_y[ind], marker='*', s=54, label=show)
	# 	plt.legend()

	f, p = signal.welch(data)
	f, p = f[1:], [np.real(pi) for pi in p[1:]]
	# plt.plot(f, p, linewidth=.5)
	# plt.plot(f, p, label=k)
	# thresh_top = np.percentile(p, 90)
	# res_k, _ = signal.find_peaks(p, height=thresh_top, prominence=0.3)
	thresh_top = np.percentile(p, 95)
	res_k, _ = signal.find_peaks(p, height=thresh_top)
	res_ks = np.around(1 / f[res_k]).astype(int)
	# plotmax(res_k, f, p)
	# plt.xlabel('Frequency')
	# plt.ylabel('PSD')
	# plt.title(k)
	a1, b1 = optimize.curve_fit(linear, np.log10(f), np.log10(p))[0]
	# x1 = np.log10(f)
	# y1 = a1 * x1 + b1
	a0 = format(a1, '.2f')
	# plt.loglog(f, 10 ** y1, label=str(a0))
	# plt.semilogy(f, 10 ** y1, label=str(a0))
	# plt.legend(loc='best')
	# plt.tight_layout()
	# plt.show()
	print(f'beta is:{a0}')
	print(f'period is:{res_ks}')
	return res_ks


def get_period_periodogram(data):
	def linear(x, a, b):
		return a * x + b

	# def plotmax(arr, arr_x, arr_y):
	# 	for ind in arr:
	# 		show = '(' + str(round(1 / arr_x[ind], 3)) + ')'
	# 		plt.scatter(arr_x[ind], arr_y[ind], marker='*', s=54, label=show)
	# 	plt.legend()

	f, p = signal.periodogram(data)
	f, p = f[1:], p[1:]
	# plt.plot(f, p, linewidth=.5)
	# plt.plot(f, p, label=k)
	thresh_top = np.mean(np.log10(p)) + 3 * np.std(np.log10(p))  # 3 sigma原理
	res_k, _ = signal.find_peaks(np.log10(p), height=thresh_top, prominence=3)
	res_ks = np.around(1 / f[res_k]).astype(int)
	# plotmax(res_k, f, p)
	# plt.xlabel('Frequency')
	# plt.ylabel('PSD')
	# plt.title(k)
	a1, b1 = optimize.curve_fit(linear, np.log10(f), np.log10(p))[0]
	x1 = np.log10(f)
	# y1 = a1 * x1 + b1
	a0 = format(a1, '.2f')
	# plt.loglog(f, 10 ** y1, label=str(a0))
	# plt.semilogy(f, 10 ** y1, label=str(a0))
	# plt.legend(loc='best')
	# plt.tight_layout()
	# plt.show()
	print(f'beta is:{a0}')
	print(f'period is:{res_ks}')
	return res_ks


class auto_orders:
	def __init__(self, data, data_name, max_order, period=None, k=None, func='auto_arima_f', model=None):
		# 初始化保存所需数据
		if model == 'vv':
			self.data = data[k][data_name]
		else:
			self.data = data[data_name][k] if k is not None else data[data_name]
		self.data_name = data_name
		self.max_order = max_order
		self.period = period if period else 1
		self.seasonal = period if period else False
		self.k = k
		self.func_name = func

	def auto_arima(self):
		print(f'Searching order of p and q for : {self.data_name, self.k}')
		if self.func_name == 'auto_arima':
			stepwise_model = auto_arima(
				self.data, start_p=0, start_q=0,
				max_p=self.max_order, max_q=self.max_order, d=0, max_d=2,
				start_Q=0, start_P=0, max_P=self.max_order, max_Q=self.max_order,
				D=0, max_D=2, m=self.period, seasonal=self.seasonal,
				trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
				maxiter=1000)
			order = stepwise_model.get_params().get('order')
			seasonal_order = stepwise_model.get_params().get('seasonal_order')

		else:
			stepwise_model = auto_arima_f(
				x=self.data.values, start_p=0, start_q=0,
				max_p=self.max_order, max_q=self.max_order, d=0, max_d=2,
				start_Q=0, start_P=0, max_P=self.max_order, max_Q=self.max_order,
				D=0, max_D=2, period=self.period, seasonal=self.seasonal,
				stepwise=True, trace=True, )
			p, q, P, Q, s, d, D = stepwise_model.get('arma')
			order = (p, d, q)
			seasonal_order = (P, D, Q, s)
		print(f'optimal order for:{self.data_name} is: {order, seasonal_order} \n\n')
		if self.seasonal:
			if self.k is not None:
				return self.k, (order, seasonal_order)
			else:
				return self.data_name, (order, seasonal_order)
		else:
			if self.k is not None:
				return self.k, order
			else:
				return self.data_name, order


def get_max_order(data, k=None):
	orders = []
	if k is not None:
		for name in data.columns:
			orders += set(get_period_welch(data[name][k]))
		orders = sorted(Counter(orders).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		if not orders:
			orders_global = []
			for name in data.columns:
				orders_global += set(get_period_welch(data[name]))
			orders_global = sorted(Counter(orders_global).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
			max_order = orders_global[0][0]
		else:
			max_order = orders[0][0]
		return k, max_order
	else:
		orders_global = []
		for name in data.columns:
			orders_global += set(get_period_welch(data[name]))
		orders_global = sorted(Counter(orders_global).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		max_order = orders_global[0][0]
		return max_order


def vmd_decomp(data, name, K):
	# . some sample parameters for VMD
	alpha = 2000  # moderate bandwidth constraint
	tau = 0.  # noise-tolerance (no strict fidelity enforcement)
	# K = 10  # 3 modes
	DC = 0  # no DC part imposed
	init = 1  # initialize omegas uniformly
	tol = 1e-7

	u, u_hat, omega = VMD(data[name].values, alpha, tau, K, DC, init, tol)
	resid = data[name].values-(u.sum(axis=0))
	return name, np.vstack((u, resid.reshape(1, -1)))


def auto_order(data, K, parallel=True):
	orders = {}
	if parallel:
		n_jobs = min((1 - ps.virtual_memory().percent / 100) / 0.4, (1 - ps.cpu_percent() / 100) * 28)
		executor = Parallel(n_jobs=int(n_jobs) + 1)
		tasks = (delayed(get_max_order)(data, k=k) for k in range(K+1))
		res = executor(tasks)
		orders = dict(res)
	else:
		for name in data.columns:
			orders[name] = get_max_order(name)[1]
	return orders