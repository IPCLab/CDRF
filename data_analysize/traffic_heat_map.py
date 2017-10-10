import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import sys
sys.path.append('/home/mldp/ML_with_bigdata')
import data_utility as du
from multi_task_data import Prepare_Task_Data
root_dir = '/home/mldp/ML_with_bigdata'


def feature_scaling(input_data, scaler=None, feature_range=(0.1, 255)):
		input_shape = input_data.shape
		input_data = input_data.reshape(-1, 1)
		if scaler:
			output = scaler.transform(input_data)
		else:
			scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
			output = scaler.fit_transform(input_data)
		return output.reshape(input_shape), scaler


def fetch_data():
	TK = Prepare_Task_Data('./npy/final')
	grid_limit = [(0, 100), (0, 100)]
	X_array, _ = TK.Task_max(grid_limit, generate_data=False)  # only need 10 minutes
	return X_array


def sum_of_grid_traffic(data_array):
	print('data_array shape:{}'.format(data_array.shape))
	data_array = np.transpose(data_array, (2, 3, 0, 1, 4))
	internet_array = data_array[:, :, :, :, 2]
	internet_array, _ = feature_scaling(internet_array)  # prevent from too large value when cumulating
	# print(internet_array.shape)
	internet_array = np.sum(internet_array, axis=(2, 3))
	print(internet_array.shape)
	return internet_array


def plot_heap_map(data_array):
	data_array, _ = feature_scaling(data_array, feature_range=(0.1, 255))

	plt.imshow(data_array, vmin=0, vmax=255, cmap=plt.get_cmap('Reds'))
	plt.grid(True)
	plt.colorbar()
	plt.show()


if __name__ == "__main__":
	data_array = fetch_data()
	internet_array = sum_of_grid_traffic(data_array)
	plot_heap_map(internet_array)