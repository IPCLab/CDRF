import numpy as np
from utility import feature_scaling, root_dir
import sys
import CNN_RNN_config
from CNN_RNN import CNN_RNN_2
import os
sys.path.append(root_dir)
import data_utility as du
from multi_task_data import Prepare_Task_Data


def grid_search_CNN_RNN_2():
	TK = Prepare_Task_Data('./npy/final')
	X_array, Y_array = TK.Task_max_min_avg(generate_data=False)
	data_len = X_array.shape[0]
	X_array = X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	Y_array = Y_array[: 9 * data_len // 10, :, :, :, 2:]
	# X_array = X_array[:, :, 10:15, 10:15, :]
	Y_array = Y_array[:, :, 10:13, 10:13, :]
	X_array, scaler = feature_scaling(X_array)
	Y_array, _ = feature_scaling(Y_array, scaler)

	gridsearcg = CNN_RNN_config.GridSearch(X_array, Y_array)
	gridsearcg.random_grid_search(task_name='random_search_CNN_RNN_2')


if __name__ == '__main__':
	grid_search_CNN_RNN_2()
