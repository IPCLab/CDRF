from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import stationary_test as st
sys.path.append('/home/mldp/ML_with_bigdata')
# import .data_utility as du
import data_utility as du
from multi_task_data import Prepare_Task_Data



def get_trainina_data():
	TK = Prepare_Task_Data('./npy/final')
	_, Y_array = TK.Task_max_min_avg(grid_limit=[(40, 60), (40, 60)], generate_data=False)
	# data_len = Y_array.shape[0]
# X_array = X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	# train_array = Y_array[: 9 * data_len // 10]
	# test_array = Y_array[9 * data_len // 10 - 1:]
	return Y_array  # grid_id timestamp min avg max


class ARIMA_Model:

	def _difference(self, timeseries_data):
		diff = []
		for i in range(1, len(timeseries_data)):
			value = timeseries_data[i] - timeseries_data[i - 1]
			diff.append(value)
		return np.array(diff)

	def _inverse_difference(self, history, diff):
		inversed_list = []
		diff_len = len(diff)
		for i in range(len(diff)):
			inversed = diff[i] + history[-diff_len + i - 1]
			inversed_list.append(inversed)
			# print(history[-diff_len + i], inversed)

		return np.array(inversed_list)

	def _predict(self, coef, history):
		yhat = 0.0
		for i in range(1, len(coef) + 1):
			yhat += coef[i - 1] * history[-i]
		return yhat

	def _set_model_and_fit(self, train_data, order_):
		model = ARIMA(train_data, order=order_)
		model_fit = model.fit(disp=0)
		return model_fit


class MTL_ARIMA_Model(ARIMA_Model):
	def __init__(self, order_list):
		self.model_dict = {}
		self.model_dict['task_max'] = {}
		self.model_dict['task_min'] = {}
		self.model_dict['task_avg'] = {}

		self.order_list = order_list

	def set_MTL_data_set(self, data_min, data_avg, data_max):
		self.model_dict['task_max']['origin_data'] = data_max
		self.model_dict['task_avg']['origin_data'] = data_avg
		self.model_dict['task_min']['origin_data'] = data_min

	def __set_train_data_set(self, order):
		data_len = self.model_dict['task_max']['origin_data'].shape[0]
		if order[1]:
			self.model_dict['task_max']['data_diff'] = self._difference(self.model_dict['task_max']['origin_data'])
			self.model_dict['task_avg']['data_diff'] = self._difference(self.model_dict['task_avg']['origin_data'])
			self.model_dict['task_min']['data_diff'] = self._difference(self.model_dict['task_min']['origin_data'])

			self.model_dict['task_max']['train_data'] = self.model_dict['task_max']['data_diff'][: 9 * data_len // 10]
			self.model_dict['task_avg']['train_data'] = self.model_dict['task_avg']['data_diff'][: 9 * data_len // 10]
			self.model_dict['task_min']['train_data'] = self.model_dict['task_min']['data_diff'][: 9 * data_len // 10]

			self.model_dict['task_max']['test_data'] = self.model_dict['task_max']['data_diff'][-149:]
			self.model_dict['task_avg']['test_data'] = self.model_dict['task_avg']['data_diff'][-149:]
			self.model_dict['task_min']['test_data'] = self.model_dict['task_min']['data_diff'][-149:]

		else:
			self.model_dict['task_max']['train_data'] = self.model_dict['task_max']['origin_data'][: 9 * data_len // 10]
			self.model_dict['task_avg']['train_data'] = self.model_dict['task_avg']['origin_data'][: 9 * data_len // 10]
			self.model_dict['task_min']['train_data'] = self.model_dict['task_min']['origin_data'][: 9 * data_len // 10]

			self.model_dict['task_max']['test_data'] = self.model_dict['task_max']['origin_data'][-149:]
			self.model_dict['task_avg']['test_data'] = self.model_dict['task_avg']['origin_data'][-149:]
			self.model_dict['task_min']['test_data'] = self.model_dict['task_min']['origin_data'][-149:]

	def MTL_predict(self):
		MTL_keys = self.model_dict.keys()
		order_list_len = len(self.order_list)
		for each_key in MTL_keys:
			order_index = 0
			keep_index = 0
			best_keep_index = 0
			while True:
				order = self.order_list[order_index]
				try:
					self._run_predicit(each_key, order)

				except Exception as err:
					# print(err)
					pass
				else:
					accu = self.__evaluate_by_task(each_key)
					print('task name:{} order:{} accu:{}'.format(each_key, order, accu))
					if accu > 0.3 and accu <= 0.9:
						print(each_key, order)
						break
					if accu > 0.9:
						best_keep_index = order_index
					else:
						keep_index = order_index

				if order_list_len > order_index + 1:
					order_index += 1
				else:
					try:
						self._run_predicit(each_key, self.order_list[best_keep_index])
					except Exception:
						self._run_predicit(each_key, self.order_list[keep_index])
					break

	def _run_predicit(self, task_key, order):
		model = self.model_dict[task_key]
		if order[1]:
			self.__set_train_data_set(order)
			history = [x for x in model['train_data']]
			prediction_diff_list = []
			expected_diff_list = []
			order = (order[0], 0, order[2])
			try:
				model_fit = self._set_model_and_fit(history, order)
			except (RuntimeError, TypeError, NameError, ValueError) as err:
				raise(err)
			except:
				print('unexpected error')
				raise

			for t in range(len(model['test_data'])):
				y_hat = self._predict(model_fit.arparams, history) + self._predict(model_fit.maparams, history)
				prediction_diff_list.append(y_hat)
				obs = model['test_data'][t]
				expected_diff_list.append(obs)
				history.append(obs)

			prediction_diff = np.array(prediction_diff_list)
			expected_diff = np.array(expected_diff_list)

			fitted_diff_value = model_fit.fittedvalues

			fitted_origin_data = model['origin_data'][: 9 * model['origin_data'].shape[0] // 10 + 1]

			predictions = self._inverse_difference(model['origin_data'], prediction_diff)
			expedcted = self._inverse_difference(model['origin_data'], expected_diff)
			fitted_value = self._inverse_difference(fitted_origin_data, fitted_diff_value)
			# predictions = prediction_diff
			# expedcted = expected_diff

		else:
			self.__set_train_data_set(order)
			history = [x for x in model['train_data']]
			prediction_list = []
			expected_list = []
			try:
				model_fit = self._set_model_and_fit(history, order)
			except (RuntimeError, TypeError, NameError, ValueError) as err:
				raise(err)
			except:
				print('unexpected error')
				raise
			for t in range(len(model['test_data'])):
				y_hat = self._predict(model_fit.arparams, history) + self._predict(model_fit.maparams, history)
				prediction_list.append(y_hat)
				obs = model['test_data'][t]
				expected_list.append(obs)
				history.append(obs)
			predictions = np.array(prediction_list)
			expedcted = np.array(expected_list)
			fitted_value = model_fit.fittedvalues
		model['model_fit'] = model_fit
		model['predict'] = predictions
		model['expected'] = expedcted
		model['fitted_value'] = fitted_value
		# print(predictions.shape, expedcted.shape, ar_coef)
		# for i, prediction in enumerate(predictions):
			# print('prediciotn:{} expedcted:{}'.format(prediction, expedcted[i]))
		# print(self.order)
		return expedcted, predictions

	def evalue_fit_value(self):
		MTL_keys = self.model_dict.keys()
		data_len = self.model_dict['task_max']['origin_data'].shape[0]
		for each_key in MTL_keys:
			model = self.model_dict[each_key]
			model_fit = model['model_fit']
			data = model['origin_data'][: 9 * data_len // 10 + 1]
			plt.figure()

			fitted = self.__inverse_difference(data, model_fit.fittedvalues)
			real_value = self.__inverse_difference(data, model['train_data'])
			plt.plot(fitted[:100], label='fitted value', marker='.')
			plt.plot(real_value[:100], label='real value', marker='.')
			plt.legend()

		plt.show()

	def MAPE(self, real, predict):
		mean_real = real.mean()
		AE = np.absolute(real - predict)
		MAPE = np.divide(AE, mean_real)
		MAPE_mean = MAPE.mean()
		if MAPE_mean > 1 or MAPE_mean < 0:
			print('Error! MAPE:{} AE_mean:{} mean_real:{}'.format(MAPE_mean, AE.mean(), mean_real))
			MAPE_mean = 1
		return MAPE_mean

	def __evaluate_by_task(self, task_key):
		model = self.model_dict[task_key]
		predict = model['predict']
		expected = model['expected']
		mape = self.MAPE(expected, predict)
		return 1 - mape

	def evaluate(self):
		MTL_keys = self.model_dict.keys()
		for each_key in MTL_keys:
			model = self.model_dict[each_key]
			predict = model['predict']
			expected = model['expected']
			# for i, prediction in enumerate(predict):
				# print('prediciotn:{} expedcted:{}'.format(prediction, expected[i]))
			mape = self.MAPE(expected, predict)
			print('task name:{} accu:{}'.format(each_key, 1 - mape))
			# print(expected.shape)
			plt.figure()
			plt.plot(expected, label='expected', marker='.')
			plt.plot(predict, label='prediction', marker='.')
			plt.legend()
			plt.grid()
			plt.title(each_key)
			'''
			err = expected - predict
			df_err = pd.DataFrame(err)
			df_err.plot(kind='kde')
			'''

	def get_predict(self):
		predict_max = self.model_dict['task_max']['predict']
		predict_avg = self.model_dict['task_avg']['predict']
		predict_min = self.model_dict['task_min']['predict']

		return predict_min, predict_avg, predict_max

	def get_fitted_value(self):
		fitted_max = self.model_dict['task_max']['fitted_value']
		fitted_avg = self.model_dict['task_avg']['fitted_value']
		fitted_min = self.model_dict['task_min']['fitted_value']

		return fitted_min, fitted_avg, fitted_max


def stationat_test(timeseries_array):
	# timeseries_array = timeseries_array[1:] - timeseries_array[:-1]
	st.series_plot(timeseries_array)
	st.lag_plot(timeseries_array)
	st.Pearson_correlation_coefficient(timeseries_array)
	st.autocorrelation_plot(timeseries_array)
	# st.pcf(timeseries_array)
	st.plot_acf_and_pacf(timeseries_array)
	st.distribution_plot(timeseries_array)
	st.split_and_test_var_mean(timeseries_array)
	st.augmented_dickey_fuller_test(timeseries_array)
	# st.decompose_seasonal(timeseries_array)
	plt.legend()
	plt.show()


if __name__ == '__main__':

	data_array = get_trainina_data()
	row = 10
	col = 1
	data_array = data_array[:, 0, row, col, :]
	stationat_test(data_array[:, 3])
	# '''
	# order_list = [(3, 1, 3), (3, 1, 2), (3, 1, 1), (2, 1, 3), (2, 1, 2), (2, 1, 1), (1, 1, 3), (1, 1, 2), (1, 1, 1), (3, 1, 0), (1, 1, 0), (2, 1, 0), (3, 0, 3), (3, 0, 2), (2, 0, 2), (2, 0, 1), (3, 0, 1)]
	# order_list = [(3, 0, 3), (3, 0, 2), (2, 0, 2), (2, 0, 1), (3, 0, 1)]
	# '''
	# order_list = [(3, 1, 0)]
	# arima = MTL_ARIMA_Model(order_list)
	# arima.set_MTL_data_set(data_array[:, 2], data_array[:, 3], data_array[:, 4])
	# arima.MTL_predict()
	# arima.evaluate()
	# plt.show()
	'''
	residuals = pd.DataFrame(arima.model_fit.resid)
	residuals.plot()
	residuals.plot(kind='kde')

	print(residuals.describe())
	plt.figure()
	timeseries_array = train_array[:, 2]
	timeseries_array_diff = timeseries_array[1:] - timeseries_array[:-1]
	plt.plot(timeseries_array_diff)
	plt.plot(arima.model_fit.fittedvalues)
	'''

