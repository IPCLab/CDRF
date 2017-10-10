import numpy as np
import utility
import CNN_RNN_config
from CNN_RNN import CNN_RNN, CNN_3D, RNN
import sys
import os
sys.path.append(utility.root_dir)
import data_utility as du
from multi_task_data import Prepare_Task_Data
from auto_regression.ARIMA_model import MTL_ARIMA_Model

root_dir = utility.root_dir
grid_row_num = 100
grid_column_num = 100

logger = utility.setlog()

def get_trainina_data():
	TK = Prepare_Task_Data('./npy/final')
	X_array, Y_array = TK.Task_max_min_avg(generate_data=True)
	data_len = X_array.shape[0]
	X_array = X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	Y_array = Y_array[: 9 * data_len // 10, :, :, :, 2:]
	# X_array = X_array[:, :, 10:15, 10:15, :]
	Y_array = Y_array[:, :, 7:10, 7:10, :]
	X_array, scaler = utility.feature_scaling(X_array)
	Y_array, _ = utility.feature_scaling(Y_array, scaler)

	return X_array, Y_array


def get_prediction_data():
	TK = Prepare_Task_Data('./npy/final')
	X_array, Y_array = TK.Task_max_min_avg(generate_data=True)
	data_len = X_array.shape[0]

	X_array_train = X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	# Y_array_train = Y_array[: 9 * data_len // 10, :, :, :, 2:]
	_, scaler = utility.feature_scaling(X_array_train)
	del X_array_train

	X_array_test = X_array[9 * data_len // 10:]
	Y_array_test = Y_array[9 * data_len // 10:, :, 10:13, 10:13, :]

	# X_array_test = X_array[0:200]
	# Y_array_test = Y_array[0:200, :, 10:13, 10:13, :]

	new_X_array, _ = utility.feature_scaling(X_array_test[:, :, :, :, 2:], scaler)
	new_Y_array, _ = utility.feature_scaling(Y_array_test[:, :, :, :, 2:], scaler)
	X_info = X_array_test[:, :, :, :, :2]
	Y_info = Y_array_test[:, :, :, :, :2]
	X_array_test = np.concatenate((X_info, new_X_array), axis=-1)
	Y_array_test = np.concatenate((Y_info, new_Y_array), axis=-1)

	# X_array = copy(X_array, new_X_array)
	# Y_array = copy(Y_array, new_Y_array)
	# print(X_array.shape)
	return X_array_test, Y_array_test, scaler


def get_all_data():
	TK = Prepare_Task_Data('./npy/Milano')
	# X_array, Y_array = TK.Task_max_min_avg(grid_limit=[(30, 90), (20, 80)], generate_data=True)
	X_array, Y_array = TK.Task_max_min_avg(grid_limit=[(0, 100), (0, 100)], generate_data=False)
	data_len = X_array.shape[0]

	X_train = X_array[: 9 * data_len // 10]
	Y_train = Y_array[: 9 * data_len // 10]
	X_test = X_array[9 * data_len // 10:]
	Y_test = Y_array[9 * data_len // 10:]
	'''
	X_train_info = X_array[: 9 * data_len // 10, :, :, :, :2]
	X_train = X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	Y_train_info = Y_array[: 9 * data_len // 10, :, :, :, :2]
	Y_train = Y_array[: 9 * data_len // 10, :, :, :, 2:]

	X_test_info = X_array[9 * data_len // 10:, :, :, :, :2]
	X_test = X_array[9 * data_len // 10:, :, :, :, -1, np.newaxis]
	Y_test_info = Y_array[9 * data_len // 10:, :, :, :, :2]
	Y_test = Y_array[9 * data_len // 10:, :, :, :, 2:]

	X_train, scaler = utility.feature_scaling(X_train)
	Y_train, _ = utility.feature_scaling(Y_train, scaler)

	X_test, _ = utility.feature_scaling(X_test, scaler)
	Y_test, _ = utility.feature_scaling(Y_test, scaler)

	X_train = np.concatenate((X_train_info, X_train), axis=-1)
	Y_train = np.concatenate((Y_train_info, Y_train), axis=-1)

	X_test = np.concatenate((X_test_info, X_test), axis=-1)
	Y_test = np.concatenate((Y_test_info, Y_test), axis=-1)
	'''
	return X_train, Y_train, X_test, Y_test


def get_network_input_and_output(X_all_train, Y_all_train, X_all_test, Y_all_test, row_range, col_range):
	# print(X_all_train.shape, Y_all_train.shape, X_all_test.shape, Y_all_test.shape)
	X_train_info = X_all_train[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], :2]
	X_train = X_all_train[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], -1, np.newaxis]
	Y_train_info = Y_all_train[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], :2]
	Y_train = Y_all_train[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], 2:]

	X_test_info = X_all_test[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], :2]
	X_test = X_all_test[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], -1, np.newaxis]
	Y_test_info = Y_all_test[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], :2]
	Y_test = Y_all_test[:, :, row_range[0]:row_range[1], col_range[0]:col_range[1], 2:]

	X_train, scaler = utility.feature_scaling(X_train)
	Y_train, _ = utility.feature_scaling(Y_train, scaler)

	X_test, _ = utility.feature_scaling(X_test, scaler)
	Y_test, _ = utility.feature_scaling(Y_test, scaler)

	X_train = np.concatenate((X_train_info, X_train), axis=-1)
	Y_train = np.concatenate((Y_train_info, Y_train), axis=-1)

	X_test = np.concatenate((X_test_info, X_test), axis=-1)
	Y_test = np.concatenate((Y_test_info, Y_test), axis=-1)

	Y_train = Y_train[:, :, 7:10, 7:10]
	Y_test = Y_test[:, :, 7:10, 7:10]
	# print(row_range, col_range, X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

	return X_train, Y_train, X_test, Y_test, scaler


def train_CNN_3D(X_array, Y_array):
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	result_path = './result/CNN_3D/'
	utility.check_path_exist(result_path)
	model_path = {
		'reload_path': './output_model/CNN_3D_all.ckpt',
		'save_path': './output_model/CNN_3D_all.ckpt',
		'result_path': result_path
	}
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	# hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
	neural = CNN_3D(input_data_shape, output_data_shape, hyper_config)
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 0, np.newaxis], 'min_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
	del X_array, Y_array
	neural.start_MTL_train(model_path, reload=False)
	return neural


def train_RNN(X_array, Y_array):
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	result_path = './result/RNN/'
	utility.check_path_exist(result_path)
	model_path = {
		'reload_path': './output_model/RNN_all.ckpt',
		'save_path': './output_model/RNN_all.ckpt',
		'result_path': result_path
	}
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	# hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
	neural = RNN(input_data_shape, output_data_shape, hyper_config)
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 0, np.newaxis], 'min_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
	del X_array, Y_array
	neural.start_MTL_train(model_path, reload=False)
	return neural


def train_CNN_RNN(X_array, Y_array):
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	result_path = './result/CNN_RNN_STL/'
	utility.check_path_exist(result_path)
	model_path = {
		'reload_path': './output_model/CNN_RNN_all.ckpt',
		'save_path': './output_model/CNN_RNN_all.ckpt',
		'result_path': result_path
	}
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
	hyper_config.iter_epoch = 1000
	neural = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 0, np.newaxis], 'min_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
	del X_array, Y_array
	neural.start_MTL_train(model_path, reload=False)
	return neural


def train_STL_CNN_RNN_(X_array, Y_array):
	task_index = {
		'min_traffic': 0,
		'avg_traffic': 1,
		'max_traffic': 2
	}

	def run_task(X_array, Y_array, task_name):
		input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
		output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
		result_path = './result/CNN_RNN'
		result_path = os.path.join(result_path, task_name)
		reload_path = '/home/mldp/ML_with_bigdata/CNN_RNN/output_model/' + 'CNN_RNN_' + str(task_name) + '.ckpt'

		utility.check_path_exist(result_path)
		model_path = {
			'reload_path': reload_path,
			'save_path': reload_path,
			'result_path': result_path
		}
		hyper_config = CNN_RNN_config.HyperParameterConfig()
		hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
		hyper_config.iter_epoch = 500
		neural = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, task_index[task_name], np.newaxis], task_name)
		# neural.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
		# neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
		del X_array, Y_array
		neural.start_STL_train(model_path, task_name, reload=False)
		return neural

	run_task(X_array, Y_array, 'max_traffic')
	run_task(X_array, Y_array, 'avg_traffic')
	run_task(X_array, Y_array, 'min_traffic')


def train_CNN_RNN_without_task(X_array, Y_array):
	logger.debug('X_array:{} Y_array:{}'.format(X_array.shape, Y_array.shape))
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	result_path = '.result/CNN_RNN_without_task'
	# result_path = os.path.join(result_path, task_name)
	reload_path = './output_model/' + 'CNN_RNN_without_task.ckpt'

	utility.check_path_exist(result_path)
	model_path = {
		'reload_path': reload_path,
		'save_path': reload_path,
		'result_path': result_path
	}
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
	hyper_config.iter_epoch = 500
	neural = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
	neural.create_MTL_task(X_array, Y_array, '_traffic')
	# neural.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
	# neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
	del X_array, Y_array
	neural.start_STL_train(model_path, '_traffic', reload=False)
	return neural


def predict_(neural, X_array, Y_array, model_path, batch_size):

	def predict_MTL_train(neural, X_array, Y_array, model_path):
		prediction_min, prediction_avg, prediction_max = neural.start_MTL_predict(
			X_array,
			Y_array,
			model_path)
		prediction_y = np.concatenate(
			(prediction_min, prediction_avg, prediction_max), axis=-1)
		return prediction_y

	def predict_remainder(neural, X_array, Y_array, model_path, batch_size):
		array_len = X_array.shape[0]
		n_remain = array_len % batch_size
		new_x = X_array[array_len - batch_size:]
		new_y = Y_array[array_len - batch_size:]
		print('new_x, new_y', new_x.shape, new_y.shape)
		prediction = predict_MTL_train(neural, new_x, new_y, model_path)
		remainder_prediction = prediction[prediction.shape[0] - n_remain:]
		return remainder_prediction

	y_prediction = predict_MTL_train(neural, X_array, Y_array, model_path)
	# print(y_prediction.shape, Y_array.shape)
	remain_predcition = predict_remainder(neural, X_array, Y_array, model_path, batch_size)
	y_prediction = np.concatenate((y_prediction, remain_predcition), axis=0)

	return y_prediction


def predict_RNN(X_array, Y_array, scaler, Neural=None):
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], 1]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	hyper_config = CNN_RNN_config.HyperParameterConfig()

	key_var = hyper_config.get_variable()
	batch_size = key_var['batch_size']
	model_path = {
		'reload_path': './output_model/RNN_all.ckpt',
		'save_path': './output_model/RNN_all.ckpt'
	}
	if Neural:
		neural = Neural
	else:
		neural = RNN(input_data_shape, output_data_shape, hyper_config)
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'min_traffic')
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, 3, np.newaxis], 'avg_traffic')
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, 4, np.newaxis], 'max_traffic')
	Y_info = Y_array[:, :, :, :, :2]
	Y_real = Y_array[:, :, :, :, 2:]

	prediction_y = predict_(neural, X_array[:, :, :, :, 2:], Y_real, model_path, batch_size)
	prediction_y = utility.un_feature_scaling(prediction_y, scaler)
	Y_real = utility.un_feature_scaling(Y_real, scaler)
	Y_array = np.concatenate((Y_info, Y_real, prediction_y), axis=-1)
	# du.save_array(Y_array, './result/RNN/Y_real_prediction.npy')
	# plot_func.plot_predict_vs_real(Y_array[:, 0, 2, 1, :2], Y_array[:, 0, 2, 1, 2], Y_array[:, 0, 2, 1, 5], 'Min_task')
	# plt.show()
	return Y_array


def predict_3D_CNN(X_array, Y_array, scaler, Neural=None):
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], 1]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	hyper_config = CNN_RNN_config.HyperParameterConfig()

	key_var = hyper_config.get_variable()
	batch_size = key_var['batch_size']
	model_path = {
		'reload_path': './output_model/CNN_3D_all.ckpt',
		'save_path': './output_model/CNN_3D_all.ckpt'
	}
	if Neural:
		neural = Neural
	else:
		neural = CNN_3D(input_data_shape, output_data_shape, hyper_config)
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'min_traffic')
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, 3, np.newaxis], 'avg_traffic')
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, 4, np.newaxis], 'max_traffic')
	Y_info = Y_array[:, :, :, :, :2]
	Y_real = Y_array[:, :, :, :, 2:]

	prediction_y = predict_(neural, X_array[:, :, :, :, 2:], Y_real, model_path, batch_size)
	prediction_y = utility.un_feature_scaling(prediction_y, scaler)
	Y_real = utility.un_feature_scaling(Y_real, scaler)
	Y_array = np.concatenate((Y_info, Y_real, prediction_y), axis=-1)

	# du.save_array(Y_array, './result/CNN_3D/Y_real_prediction.npy')

	return Y_array


def predict_CNN_RNN(X_array, Y_array, scaler, Neural=None):
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], 1]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	hyper_config.read_config(file_path=os.path.join(
		root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))

	key_var = hyper_config.get_variable()
	batch_size = key_var['batch_size']
	model_path = {
		'reload_path': './output_model/CNN_RNN_without_task.ckpt',
		'save_path': './output_model/CNN_RNN_without_task.ckpt'
	}
	if Neural:
		neural = Neural
	else:
		neural = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, -1, np.newaxis], '_traffic')

	Y_info = Y_array[:, :, :, :, :2]
	Y_real = Y_array[:, :, :, :, 2:]

	prediction_y = predict_(neural, X_array[:, :, :, :, 2:], Y_real, model_path, batch_size)
	prediction_y = utility.un_feature_scaling(prediction_y, scaler)
	Y_real = utility.un_feature_scaling(Y_real, scaler)
	Y_array = np.concatenate((Y_info, Y_real, prediction_y), axis=-1)

	# du.save_array(Y_array, './result/CNN_RNN/Y_real_prediction.npy')

	return Y_array


def predict_CNN_RNN_without_task(X_array, Y_array, scaler):
	def run_task(X_array, Y_array, task_name, scaler):
		def STL_predict(neural, X_array, Y_array, model_path, batch_size, task_name):
			def predict_remainder(neural, X_array, Y_array, model_path, batch_size, task_name):
				array_len = X_array.shape[0]
				n_remain = array_len % batch_size
				new_x = X_array[array_len - batch_size:]
				new_y = Y_array[array_len - batch_size:]
				# print('new_x, new_y', new_x.shape, new_y.shape)
				prediction = neural.start_STL_predict(new_x, new_y, model_path, task_name)
				remainder_prediction = prediction[prediction.shape[0] - n_remain:]
				return remainder_prediction
			# print(X_array.shape, Y_array.shape)
			y_prediction = neural.start_STL_predict(X_array, Y_array, model_path, task_name)
			# print(y_prediction.shape, Y_array.shape)
			remain_predcition = predict_remainder(neural, X_array, Y_array, model_path, batch_size, task_name)
			y_prediction = np.concatenate((y_prediction, remain_predcition), axis=0)

			return y_prediction
		hyper_config = CNN_RNN_config.HyperParameterConfig()
		hyper_config.read_config(file_path=os.path.join(
			root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
		key_var = hyper_config.get_variable()
		batch_size = key_var['batch_size']

		reload_path = './output_model/' + 'CNN_RNN_without_task.ckpt'
		model_path = {
			'reload_path': reload_path,
			'save_path': reload_path
		}

		input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], 1]
		output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]

		neural = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, -1, np.newaxis], task_name)
		# neural.create_MTL_task(X_array, Y_array[:, :, :, :, task_index['avg_traffic'], np.newaxis], 'avg_traffic')
		# neural.create_MTL_task(X_array, Y_array[:, :, :, :, task_index['max_traffic'], np.newaxis], 'max_traffic')

		y_task_real = Y_array[:, :, :, :, -1, np.newaxis]
		prediction_y = STL_predict(neural, X_array, y_task_real, model_path, batch_size, task_name)
		prediction_y = utility.un_feature_scaling(prediction_y, scaler)
		y_task_real = utility.un_feature_scaling(y_task_real, scaler)
		Y_task_real_prediction = np.concatenate((y_task_real, prediction_y), axis=-1)
		return Y_task_real_prediction

	Y_real_prediction = run_task(X_array[:, :, :, :, 2:], Y_array[:, :, :, :, 2:], '_traffic', scaler)
	logger.debug('Y_real_prediction shape {}'.format(Y_real_prediction.shape))
	Y_info = Y_array[:, :, :, :, :2]
	Y_array = np.concatenate((Y_info, Y_real_prediction), axis=-1)
	logger.debug('Y_array shape {}'.format(Y_array.shape))
	return Y_array


def predict_STL_CNN_RNN(X_array, Y_array, scaler):
	task_index = {
		'min_traffic': 0,
		'avg_traffic': 1,
		'max_traffic': 2
	}

	def run_task(X_array, Y_array, task_name, scaler):
		def STL_predict(neural, X_array, Y_array, model_path, batch_size, task_name):
			def predict_remainder(neural, X_array, Y_array, model_path, batch_size, task_name):
				array_len = X_array.shape[0]
				n_remain = array_len % batch_size
				new_x = X_array[array_len - batch_size:]
				new_y = Y_array[array_len - batch_size:]
				# print('new_x, new_y', new_x.shape, new_y.shape)
				prediction = neural.start_STL_predict(new_x, new_y, model_path, task_name)
				remainder_prediction = prediction[prediction.shape[0] - n_remain:]
				return remainder_prediction
			# print(X_array.shape, Y_array.shape)
			y_prediction = neural.start_STL_predict(X_array, Y_array, model_path, task_name)
			# print(y_prediction.shape, Y_array.shape)
			remain_predcition = predict_remainder(neural, X_array, Y_array, model_path, batch_size, task_name)
			y_prediction = np.concatenate((y_prediction, remain_predcition), axis=0)

			return y_prediction
		hyper_config = CNN_RNN_config.HyperParameterConfig()
		hyper_config.read_config(file_path=os.path.join(
			root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
		key_var = hyper_config.get_variable()
		batch_size = key_var['batch_size']

		reload_path = './output_model/' + 'CNN_RNN_' + str(task_name) + '.ckpt'
		model_path = {
			'reload_path': reload_path,
			'save_path': reload_path
		}

		input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], 1]
		output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]

		neural = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
		neural.create_MTL_task(X_array, Y_array[:, :, :, :, task_index[task_name], np.newaxis], task_name)
		# neural.create_MTL_task(X_array, Y_array[:, :, :, :, task_index['avg_traffic'], np.newaxis], 'avg_traffic')
		# neural.create_MTL_task(X_array, Y_array[:, :, :, :, task_index['max_traffic'], np.newaxis], 'max_traffic')

		y_task_real = Y_array[:, :, :, :, task_index[task_name], np.newaxis]
		prediction_y = STL_predict(neural, X_array, y_task_real, model_path, batch_size, task_name)
		prediction_y = utility.un_feature_scaling(prediction_y, scaler)
		y_task_real = utility.un_feature_scaling(y_task_real, scaler)
		Y_task_real_prediction = np.concatenate((y_task_real, prediction_y), axis=-1)
		return Y_task_real_prediction
	# print('X', X_array.shape)
	min_Y_real_prediction = run_task(X_array[:, :, :, :, 2:], Y_array[:, :, :, :, 2:], 'min_traffic', scaler)
	avg_Y_real_prediction = run_task(X_array[:, :, :, :, 2:], Y_array[:, :, :, :, 2:], 'avg_traffic', scaler)
	max_Y_real_prediction = run_task(X_array[:, :, :, :, 2:], Y_array[:, :, :, :, 2:], 'max_traffic', scaler)
	# print(min_Y_real_prediction.shape, avg_Y_real_prediction.shape, max_Y_real_prediction.shape)
	Y_info = Y_array[:, :, :, :, :2]
	# print(Y_info.shape)
	Y_real_prediction = np.concatenate((
		Y_info,
		min_Y_real_prediction[:, :, :, :, 0, np.newaxis],
		avg_Y_real_prediction[:, :, :, :, 0, np.newaxis],
		max_Y_real_prediction[:, :, :, :, 0, np.newaxis],
		min_Y_real_prediction[:, :, :, :, 1, np.newaxis],
		avg_Y_real_prediction[:, :, :, :, 1, np.newaxis],
		max_Y_real_prediction[:, :, :, :, 1, np.newaxis]), axis=-1)
	print('Y_real_prediction', Y_real_prediction.shape)
	return Y_real_prediction


def predict_ARIMA(input_array):
	# print(input_array.shape)
	data_info = input_array[:, :2]
	data_array = input_array[:, 2:]

	order_list = [(3, 1, 3), (3, 1, 2), (3, 1, 1), (2, 1, 3), (2, 1, 2), (2, 1, 1), (1, 1, 3), (1, 1, 2), (1, 1, 1), (3, 0, 3), (3, 0, 0), (3, 0, 2), (2, 0, 3), (2, 0, 1), (1, 1, 0), (2, 1, 0), (2, 0, 0), (0, 0, 3), (1, 0, 0), (0, 0, 2), (0, 0, 1)]
	arima = MTL_ARIMA_Model(order_list)
	arima.set_MTL_data_set(data_array[:, 0], data_array[:, 1], data_array[:, 2])
	arima.MTL_predict()

	preidction_min, prediction_avg, prediction_max = arima.get_predict()
	# print(preidction_min.shape, prediction_avg.shape, prediction_max.shape)
	fitted_min, fitted_avg, fitted_max = arima.get_fitted_value()
	# print(fitted_min.shape, fitted_avg.shape, fitted_max.shape)
	fitted_value = np.stack((fitted_min, fitted_avg, fitted_max), axis=-1)
	predict_value = np.stack((preidction_min, prediction_avg, prediction_max), axis=-1)
	info_fitted_predict = np.concatenate((fitted_value, predict_value), axis=0)
	len_info_fitted_predict = info_fitted_predict.shape[0]
	info_fitted_predict = np.concatenate((data_info[-len_info_fitted_predict:], data_array[-len_info_fitted_predict:], info_fitted_predict), axis=-1)
	# print(info_fitted_predict.shape)

	return info_fitted_predict


def compute_range_by_center(row_center, col_center):
	row_range = (row_center - 7, row_center + 7 + 1)
	col_range = (col_center - 7, col_center + 7 + 1)
	# print(row_range, col_range)
	return row_range, col_range


def store_report(Y_array, file_path, row_center, col_center):
	data_len = Y_array.shape[0]
	MAPE_test = utility.MAPE_loss(Y_array[9 * data_len // 10:, :, :, :, 2, np.newaxis], Y_array[9 * data_len // 10:, :, :, :, 3, np.newaxis])
	if MAPE_test:
		print('accu', 1 - MAPE_test)
		string_ = 'row center:{} col center:{} accu:{:.4f}\n'.format(row_center, col_center, 1 - MAPE_test)
		print(string_)
		with open(file_path, 'a',) as f:
			f.write(string_)
	
	# # min_MAPE_test = utility.MAPE_loss(Y_array[9 * data_len // 10:, :, :, :, 2, np.newaxis], Y_array[9 * data_len // 10:, :, :, :, 5, np.newaxis])
	# # avg_MAPE_test = utility.MAPE_loss(Y_array[9 * data_len // 10:, :, :, :, 3, np.newaxis], Y_array[9 * data_len // 10:, :, :, :, 6, np.newaxis])
	# # max_MAPE_test = utility.MAPE_loss(Y_array[9 * data_len // 10:, :, :, :, 4, np.newaxis], Y_array[9 * data_len // 10:, :, :, :, 7, np.newaxis])
	# if min_MAPE_test and avg_MAPE_test and max_MAPE_test:
	# 	print('min accu', 1 - min_MAPE_test)
	# 	print('avg accu', 1 - avg_MAPE_test)
	# 	print('max accu', 1 - max_MAPE_test)
	# 	string_ = 'row center:{} col center:{} min accu:{:.4f} avg accu:{:.4f} max accu:{:.4f}\n'.format(row_center, col_center, 1 - min_MAPE_test, 1 - avg_MAPE_test, 1 - max_MAPE_test)
	# 	print(string_)
	# 	with open(file_path, 'a',) as f:
	# 		f.write(string_)


def loop_CNN_RNN():
	X_all_train, Y_all_train, X_all_test, Y_all_test = get_all_data()
	# print(X_all_train.shape, Y_all_train.shape, X_all_test.shape, Y_all_test.shape)
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	'''
	gird_id, timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max
	'''
	traffic_array = np.zeros([1487, 1, grid_row_num, grid_column_num, 8], dtype=float)

	base_dir = os.path.join(root_dir, 'CNN_RNN/result/CNN_RNN')
	store_path = os.path.join(base_dir, 'loop_report.txt')
	if os.path.exists(store_path):
		os.remove(store_path)
	for row_center in row_center_list:
		for col_center in col_center_list:
			print('row_center:{} col_center:{}'.format(row_center, col_center))
			row_range, col_range = compute_range_by_center(row_center, col_center)
			X_train, Y_train, X_test, Y_test, scaler = get_network_input_and_output(X_all_train, Y_all_train, X_all_test, Y_all_test, row_range, col_range)
			neural = train_CNN_RNN(X_train[:, :, :, :, -1, np.newaxis], Y_train[:, :, :, :, 2:])

			X_all = np.concatenate((X_train, X_test), axis=0)
			Y_all = np.concatenate((Y_train, Y_test), axis=0)
			traffic_data = predict_CNN_RNN(X_all, Y_all, scaler, neural)
			store_report(traffic_data, store_path, row_center, col_center)
			for i in range(traffic_data.shape[0]):
				for j in range(traffic_data.shape[1]):
					for row in range(traffic_data.shape[2]):
						for col in range(traffic_data.shape[3]):
							grid_id = traffic_data[i, j, row, col, 0]
							row_index, col_index = utility.compute_row_col(grid_id)
							# print(row_index, col_index)
							traffic_array[i, j, row_index, col_index] = traffic_data[
								i, j, row, col]
							# print(traffic_array[i, j, row_index, col_index])

		du.save_array(traffic_array, os.path.join(base_dir, 'all_real_prediction_traffic_array.npy'))
	du.save_array(traffic_array, os.path.join(base_dir, 'all_real_prediction_traffic_array.npy'))


def loop_CNN_3D():
	X_all_train, Y_all_train, X_all_test, Y_all_test = get_all_data()
	# print(X_all_train.shape, Y_all_train.shape, X_all_test.shape, Y_all_test.shape)
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	'''
	gird_id, timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max
	'''
	traffic_array = np.zeros([1487, 1, grid_row_num, grid_column_num, 8], dtype=float)
	store_path = os.path.join('./result/CNN_3D', 'loop_report.txt')
	os.remove(store_path)
	for row_center in row_center_list:
		for col_center in col_center_list:
			print('row_center:{} col_center:{}'.format(row_center, col_center))
			row_range, col_range = compute_range_by_center(row_center, col_center)

			X_train, Y_train, X_test, Y_test, scaler = get_network_input_and_output(X_all_train, Y_all_train, X_all_test, Y_all_test, row_range, col_range)
			neural = train_CNN_3D(X_train[:, :, :, :, -1, np.newaxis], Y_train[:, :, :, :, 2:])
			X_all = np.concatenate((X_train, X_test), axis=0)
			Y_all = np.concatenate((Y_train, Y_test), axis=0)
			traffic_data = predict_3D_CNN(X_all, Y_all, scaler, neural)
			store_report(traffic_data, store_path, row_center, col_center)
			# print(traffic_data.shape)
			# print(X_test.shape)
			# print(traffic_array[2000])
			# '''
			for i in range(traffic_data.shape[0]):
				for j in range(traffic_data.shape[1]):
					for row in range(traffic_data.shape[2]):
						for col in range(traffic_data.shape[3]):
							grid_id = traffic_data[i, j, row, col, 0]
							row_index, col_index = utility.compute_row_col(grid_id)
							# print(row_index, col_index)
							traffic_array[i, j, row_index, col_index] = traffic_data[
								i, j, row, col]
							# print(traffic_array[i, j, row_index, col_index])

		du.save_array(traffic_array, './result/CNN_3D/all_real_prediction_traffic_array.npy')
	du.save_array(traffic_array, './result/CNN_3D/all_real_prediction_traffic_array.npy')


def loop_RNN():
	X_all_train, Y_all_train, X_all_test, Y_all_test = get_all_data()
	# print(X_all_train.shape, Y_all_train.shape, X_all_test.shape, Y_all_test.shape)
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	'''
	gird_id, timestamp, real_min, real_avg, real_max, prediction_min, prediction_avg, prediction_max
	'''
	traffic_array = np.zeros([1487, 1, grid_row_num, grid_column_num, 8], dtype=float)
	store_path = os.path.join('./result/RNN', 'loop_report.txt')
	os.remove(store_path)
	for row_center in row_center_list:
		for col_center in col_center_list:
			print('row_center:{} col_center:{}'.format(row_center, col_center))
			row_range, col_range = compute_range_by_center(row_center, col_center)
			X_train, Y_train, X_test, Y_test, scaler = get_network_input_and_output(X_all_train, Y_all_train, X_all_test, Y_all_test, row_range, col_range)
			neural = train_RNN(X_train[:, :, :, :, -1, np.newaxis], Y_train[:, :, :, :, 2:])
			X_all = np.concatenate((X_train, X_test), axis=0)
			Y_all = np.concatenate((Y_train, Y_test), axis=0)
			traffic_data = predict_RNN(X_all, Y_all, scaler, neural)

			store_report(traffic_data, store_path, row_center, col_center)
			for i in range(traffic_data.shape[0]):
				for j in range(traffic_data.shape[1]):
					for row in range(traffic_data.shape[2]):
						for col in range(traffic_data.shape[3]):
							grid_id = traffic_data[i, j, row, col, 0]
							row_index, col_index = utility.compute_row_col(grid_id)
							# print(row_index, col_index)
							traffic_array[i, j, row_index, col_index] = traffic_data[
								i, j, row, col]
							# print(traffic_array[i, j, row_index, col_index])

		du.save_array(traffic_array, './result/RNN/all_real_prediction_traffic_array.npy')
	du.save_array(traffic_array, './result/RNN/all_real_prediction_traffic_array.npy')


def loop_ARIMA():
	_, Y_all_train, _, Y_all_test = get_all_data()
	Y_total = np.concatenate((Y_all_train, Y_all_test), axis=0)
	Y_total = Y_total.astype(np.float)
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	row_range = range(row_center_list[0], row_center_list[-1] + 1)
	col_range = range(col_center_list[0], col_center_list[-1] + 1)

	traffic_array = np.zeros([1487, 1, grid_row_num, grid_column_num, 8], dtype=float)
	store_path = os.path.join('./result/ARIMA', 'loop_report.txt')
	utility.check_path_exist(os.path.join(root_dir, 'CNN_RNN/result/ARIMA'))
	if os.path.exists(store_path):
		os.remove(store_path)

	for row in row_range:
		for col in col_range:
			print('row:{} col:{}'.format(row, col))
			info_fitted_predict = predict_ARIMA(Y_total[:, 0, row, col, :])
			store_report(info_fitted_predict.reshape(info_fitted_predict.shape[0], 1, 1, 1, info_fitted_predict.shape[1]), store_path, row, col)

			for i in range(1, info_fitted_predict.shape[0]):
				grid_id = info_fitted_predict[i, 0]
				row_index, col_index = utility.compute_row_col(grid_id)
				# print(row_index, col_index)
				traffic_array[i, 0, row_index, col_index] = info_fitted_predict[i]

		du.save_array(traffic_array, './result/ARIMA/all_real_prediction_traffic_array.npy')
	du.save_array(traffic_array, './result/ARIMA/all_real_prediction_traffic_array.npy')


def loop_STL_CNN_RNN():
	X_all_train, Y_all_train, X_all_test, Y_all_test = get_all_data()
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	traffic_array = np.zeros([1487, 1, grid_row_num, grid_column_num, 8], dtype=float)
	base_dir = os.path.join('./result/CNN_RNN_STL')
	store_path = os.path.join(base_dir, 'loop_report.txt')
	utility.check_path_exist(base_dir)
	if os.path.exists(store_path):
		os.remove(store_path)
	for row_center in row_center_list:
		for col_center in col_center_list:
			print('row_center:{} col_center:{}'.format(row_center, col_center))
			row_range, col_range = compute_range_by_center(row_center, col_center)
			X_train, Y_train, X_test, Y_test, scaler = get_network_input_and_output(X_all_train, Y_all_train, X_all_test, Y_all_test, row_range, col_range)
			train_STL_CNN_RNN_(X_train[:, :, :, :, -1, np.newaxis], Y_train[:, :, :, :, 2:])

			X_all = np.concatenate((X_train, X_test), axis=0)
			Y_all = np.concatenate((Y_train, Y_test), axis=0)
			traffic_data = predict_STL_CNN_RNN(X_all, Y_all, scaler)
			store_report(traffic_data, store_path, row_center, col_center)
			for i in range(traffic_data.shape[0]):
				for j in range(traffic_data.shape[1]):
					for row in range(traffic_data.shape[2]):
						for col in range(traffic_data.shape[3]):
							grid_id = traffic_data[i, j, row, col, 0]
							row_index, col_index = utility.compute_row_col(grid_id)
							# print(row_index, col_index)
							traffic_array[i, j, row_index, col_index] = traffic_data[
								i, j, row, col]
							# print(traffic_array[i, j, row_index, col_index])

		du.save_array(traffic_array, os.path.join(base_dir, 'all_real_prediction_traffic_array.npy'))
	du.save_array(traffic_array, os.path.join(base_dir, 'all_real_prediction_traffic_array.npy'))


def loop_CNN_RNN_without_task():
	def generate_tain_and_test():
		X_all_train, _, X_all_test, _ = get_all_data()
		all_data = np.concatenate((X_all_train, X_all_test), axis=0)
		del X_all_train, X_all_test
		X_all = all_data[:-1]
		Y_all = all_data[1:]
		del all_data
		data_len = X_all.shape[0]
		X_all_train, X_all_test = X_all[: 9 * data_len // 10], X_all[9 * data_len // 10:]
		Y_all_train, Y_all_test = Y_all[: 9 * data_len // 10], Y_all[9 * data_len // 10:]
		return X_all_train, Y_all_train, X_all_test, Y_all_test

	X_all_train, Y_all_train, X_all_test, Y_all_test = generate_tain_and_test()
	logger.debug('{} {} {} {}'.format(X_all_train.shape, Y_all_train.shape, X_all_test.shape, Y_all_test.shape))
	row_center_list = list(range(40, 80, 3))
	col_center_list = list(range(30, 70, 3))
	row_shift = row_center_list[0]
	col_shift = col_center_list[0]
	traffic_array = np.zeros([1487, 6, grid_row_num - row_shift - 10, grid_column_num - col_shift - 20, 4], dtype=float)

	base_dir = os.path.join('./result/CNN_RNN_without_task')
	store_path = os.path.join(base_dir, 'loop_report.txt')
	utility.check_path_exist(base_dir)
	if os.path.exists(store_path):
		os.remove(store_path)
	for row_center in row_center_list:
		for col_center in col_center_list:
			print('row_center:{} col_center:{}'.format(row_center, col_center))
			row_range, col_range = compute_range_by_center(row_center, col_center)
			X_train, Y_train, X_test, Y_test, scaler = get_network_input_and_output(X_all_train, Y_all_train, X_all_test, Y_all_test, row_range, col_range)
			logger.debug('X_train:{} Y_train:{} Y_train:{} X_test:{}'.format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
			train_CNN_RNN_without_task(X_train[:, :, :, :, -1, np.newaxis], Y_train[:, :, :, :, -1, np.newaxis])

			X_all = np.concatenate((X_train, X_test), axis=0)
			Y_all = np.concatenate((Y_train, Y_test), axis=0)
			traffic_data = predict_CNN_RNN_without_task(X_all, Y_all, scaler)
			logger.debug('traffice data shape:{}'.format(traffic_data.shape))
			store_report(traffic_data, store_path, row_center, col_center)
			for i in range(traffic_data.shape[0]):
				for j in range(traffic_data.shape[1]):
					for row in range(traffic_data.shape[2]):
						for col in range(traffic_data.shape[3]):
							grid_id = traffic_data[i, j, row, col, 0]
							if grid_id == 0:
								continue
							row_index, col_index = utility.compute_row_col(grid_id)
							if row_index not in range(39, 81) or col_index not in range(29, 71):
								continue
							# print(row_index, col_index)
							# logger.debug('grid id:{} row index:{} col_index:{}'.format(grid_id, row_index, col_index))
							# logger.debug('row_center:{} col_center:{}'.format(row_center, col_center))
							traffic_array[i, j, row_index - row_shift, col_index - col_shift] = traffic_data[
								i, j, row, col]
		du.save_array(traffic_array, os.path.join(base_dir, 'all_real_prediction_traffic_array.npy'))
	du.save_array(traffic_array, os.path.join(base_dir, 'all_real_prediction_traffic_array.npy'))


def train_different_method():
	X_array, Y_array = get_trainina_data()
	# train_CNN_RNN(X_array, Y_array)
	# train_CNN_3D(X_array, Y_array)
	# train_RNN(X_array, Y_array)


def predict_different_method():
	X_array, Y_array, scaler = get_prediction_data()
	CNN_RNN_prediction = predict_CNN_RNN(X_array, Y_array, scaler)

	# CNN_3D_prediction = predict_3D_CNN(X_array, Y_array, scaler)

	# RNN_predict = predict_RNN(X_array, Y_array, scaler)


def loop_different_method():
	# loop_CNN_RNN()
	# loop_CNN_3D()
	# loop_RNN()
	# loop_ARIMA()
	# loop_STL_CNN_RNN()
	loop_CNN_RNN_without_task()


if __name__ == '__main__':
	# train_different_method()
	# predict_different_method()
	loop_different_method()
