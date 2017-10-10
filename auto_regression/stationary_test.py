import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
sys.path.append('/home/mldp/ML_with_bigdata')
# import .data_utility as du
import data_utility as du

input_file = '/home/mldp/ML_with_bigdata/npy/11_0.npy'


def split_and_test_var_mean(timeseries_array):
	# plt.figure()
	# timeseries_array = np.log(timeseries_array)
	# plt.plot(timeseries_array)
	split = len(timeseries_array) // 2
	X1, X2 = timeseries_array[:split], timeseries_array[split:]
	mean1, mean2 = X1.mean(), X2.mean()
	var1, var2 = X1.var(), X2.var()
	print('mean1:{} mean2:{}'.format(mean1, mean2))
	print('variance1:{} variance2:{}'.format(var1, var2))


def plot_KDE(data):
	data.plot(kind='kde')
	plt.title('KDE plot')
	plt.show()


def augmented_dickey_fuller_test(timeseries_array):
	result = adfuller(timeseries_array)
	print('ADF stattistic:{}'.format(result[0]))
	print('p-value:{}'.format(result[1]))
	print('critical value:')
	for key, value in result[4].items():
		print('\t{}: {}'.format(key, value))


def lag_plot(timeseries_array):
	plt.figure()
	time_series = pd.Series(timeseries_array)
	pd.tools.plotting.lag_plot(time_series)


def series_plot(timeseries_array):
	time_series = pd.Series(timeseries_array)
	rol_mean = pd.Series.rolling(time_series, window=24).mean()
	rol_std = pd.Series.rolling(time_series, window=24).std()

	# ts_log_diff = time_series - rol_mean
	# print(ts_log_diff.head(30))
	time_series.plot()
	plt.plot(rol_mean, label='Rolling Mean')
	plt.plot(rol_std, label='Rolling std')
	plt.legend()


def autocorrelation_plot(timeseries_array):
	plt.figure()
	time_series = pd.Series(timeseries_array)
	pd.tools.plotting.autocorrelation_plot(time_series)


def pcf(timeseries_array):
	time_series = pd.Series(timeseries_array)
	plot_acf(time_series, lags=31)


def distribution_plot(timeseries_array):
	plt.figure()
	time_series = pd.Series(timeseries_array)
	# print(time_series)
	time_series.hist()


def plot_acf_and_pacf(timeseries_array):
	plt.figure()
	plt.subplot(211)
	plot_acf(timeseries_array, ax=plt.gca(), lags=31)
	plt.subplot(212)
	plot_pacf(timeseries_array, ax=plt.gca(), lags=31)


def Pearson_correlation_coefficient(timeseries_array):
	df = pd.DataFrame(timeseries_array)
	df_1 = pd.concat([df.shift(1), df], axis=1)
	df_1.columns = ['t-1', 't+1']
	result = df_1.corr()
	print(result)


def decompose_seasonal(timeseries_array):
	time_series = pd.Series(timeseries_array)
	decomposition = seasonal_decompose(time_series)
	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid
	print(decomposition)
	'''
	plt.figure()
	ax_1 = plt.add_subplot(411)
	ax_2 = plt.add_subplot(412)
	ax_3 = plt.add_subplot(413)
	ax_4 = plt.add_subplot(414)
	'''



