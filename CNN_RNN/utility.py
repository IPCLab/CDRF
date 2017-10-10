from sklearn import preprocessing
from datetime import datetime
import pytz
import os
import numpy as np
import logging

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# root_dir = '~/home'


def setlog(logger_name='logger'):
	logger_name = logger_name
	logger = logging.getLogger(name=logger_name)
	logger.setLevel(logging.DEBUG)
	log_path = './log.log'
	fh = logging.FileHandler(log_path, mode='w')
	fh.setLevel(logging.WARN)
	console = logging.StreamHandler()
	console.setLevel(logging.DEBUG)

	fmt = '%(levelname)s:[%(filename)s %(funcName)s:%(lineno)d] %(message)s'
	formatter = logging.Formatter(fmt=fmt, datefmt=None)

	fh.setFormatter(formatter)
	console.setFormatter(formatter)

	logger.addHandler(fh)
	logger.addHandler(console)
	return logger


def compute_row_col(grid_id):
	grid_row_num = 100
	grid_column_num = 100
	row = 99 - int(grid_id / grid_row_num)  # row mapping to milan grid
	column = grid_id % grid_column_num - 1
	# print(row, column)
	return int(row), int(column)


def comput_grid_id(row, col):
	Y = 99 - row
	X = col + 1
	grid_id = Y * 100 + X
	return grid_id


def feature_scaling(input_datas, scaler=None, feature_range=(0.1, 255)):
	# print(input_datas.shape)
	input_shape = input_datas.shape
	input_datas = input_datas.reshape(-1, 1)
	# print(np.amin(input_datas))
	if scaler:
		output = scaler.transform(input_datas)
	else:
		scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
		output = scaler.fit_transform(input_datas)

	output = output.reshape(input_shape)
	return output, scaler


def un_feature_scaling(input_data, scaler):
	input_shape = input_data.shape
	input_data = input_data.reshape(-1, 1)
	output = scaler.inverse_transform(input_data)
	output = output.reshape(input_shape)
	return output


def set_time_zone(timestamp):
	UTC_timezone = pytz.timezone('UTC')
	Mi_timezone = pytz.timezone('Europe/Rome')
	date_time = datetime.utcfromtimestamp(float(timestamp))
	date_time = date_time.replace(tzinfo=UTC_timezone)
	date_time = date_time.astimezone(Mi_timezone)
	return date_time


def date_time_covert_to_str(date_time):
	return date_time.strftime('%m-%d %H')


def check_path_exist(path):
	if not os.path.exists(path):
		os.makedirs(path)


def AE_loss(real, predict):
	AE = np.absolute(real - predict)
	AE_mean = AE.mean()
	# print('AE:', AE_mean)
	return AE_mean


def RMSE_loss(real, predict):
	MSE = (real - predict) ** 2
	RMSE = np.sqrt(MSE.mean())
	# print('RMSE:', RMSE)
	return RMSE


def MAPE_loss(real, predict):
	mean_real = real.mean()
	AE = np.absolute(real - predict)
	MAPE = np.divide(AE, mean_real)
	MAPE_mean = MAPE.mean()
	if MAPE_mean > 1 or MAPE_mean < 0:
		print('Error! MAPE:{} AE_mean:{} mean_real:{}'.format(MAPE_mean, AE.mean(), mean_real))
		MAPE_mean = None
	return MAPE_mean


def MAPE_loss_without_real_mean(real, predict):
	mean_real = real
	AE = np.absolute(real - predict)
	MAPE = np.divide(AE, mean_real)
	MAPE_mean = MAPE.mean()
	if MAPE_mean > 1 or MAPE_mean < 0:
		print('Error! MAPE:{} AE_mean:{}'.format(MAPE_mean, AE.mean()))
		# MAPE_mean = None
	return MAPE_mean


def find_in_obj(obj, condition, path=None):
	if path is None:
		path = []

	if isinstance(obj, list):
		for index, value in enumerate(obj):
			new_path = list(path)
			new_path.append(index)
			for result in find_in_obj(value, condition, path=new_path):
				yield result
	if isinstance(obj, dict):
		for key, value in obj.items():
			new_path = list(path)
			new_path.append(key)
			for result in find_in_obj(value, condition, path=new_path):
				yield result

			if condition == key:
				new_path = list(path)
				new_path.append(key)
				yield new_path


if __name__ == "__main__":
	print(compute_row_col(5669))
