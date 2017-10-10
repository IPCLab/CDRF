import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter, getitem
from itertools import groupby
from functools import reduce
import json
import os
import report_func
import utility
import sys
sys.path.append(utility.root_dir)
import data_utility as du

logger = utility.setlog()
root_dir = utility.root_dir

# method_LM_result_path = '/home/qiuhui/processed_data/'
# method_CNN_RNN_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/CNN_RNN/Y_real_prediction.npy'
# method_CNN_3D_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/CNN_3D/Y_real_prediction.npy'
# method_RNN_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/RNN/Y_real_prediction.npy'

# plot_grid_id_list = [4258, 4456, 4457]
# time_shift = 2


# def get_CNN_RNN():
# 	CNN_RNN_prediction = du.load_array(method_CNN_RNN_result_path)
# 	info = CNN_RNN_prediction[time_shift:, :, :, :, :2]
# 	real = CNN_RNN_prediction[time_shift:, :, :, :, 2:5]
# 	CNN_RNN_prediction = CNN_RNN_prediction[time_shift:, :, :, :, 5:]
# 	return info, real, CNN_RNN_prediction


# def get_CNN_3D():
# 	CNN_3D_prediction = du.load_array(method_CNN_3D_result_path)
# 	info = CNN_3D_prediction[time_shift:, :, :, :, :2]
# 	real = CNN_3D_prediction[time_shift:, :, :, :, 2:5]
# 	CNN_3D_prediction = CNN_3D_prediction[time_shift:, :, :, :, 5:]
# 	return info, real, CNN_3D_prediction


# def get_RNN():
# 	RNN_prediction = du.load_array(method_RNN_result_path)
# 	info = RNN_prediction[time_shift:, :, :, :, :2]
# 	real = RNN_prediction[time_shift:, :, :, :, 2:5]
# 	RNN_prediction = RNN_prediction[time_shift:, :, :, :, 5:]
# 	return info, real, RNN_prediction


# def get_LM(grid_id, task_name):
# 	min_file = os.path.join(method_LM_result_path, str(grid_id) + '_' + task_name + '.npy')
# 	min_array = du.load_array(min_file)
# 	return min_array


# def plot_all_method_together(plot_dict, task_name):
# 	def get_xlabel(timestamps):
# 		xlabel_list = []
# 		for timestamp in timestamps:
# 			datetime = utility.set_time_zone(timestamp)
# 			xlabel_list.append(utility.date_time_covert_to_str(datetime))
# 		return xlabel_list

# 	xlabel_list = get_xlabel(plot_dict['info'][:, 1])
# 	x_len = len(xlabel_list)
# 	grid_id = plot_dict['info'][0, 0]
# 	fig, ax = plt.subplots(1, 1)
# 	ax.set_xlabel('time sequence')
# 	ax.set_ylabel('number of CDR')

# 	ax.plot(range(x_len), plot_dict['real'], label='Real', color='k')
# 	ax.plot(range(x_len), plot_dict['CNN_RNN'], label='CNN_RNN', marker='.')
# 	ax.plot(range(x_len), plot_dict['CNN_3D'], label='CNN_3D', linestyle='--')
# 	ax.plot(range(x_len), plot_dict['RNN'], label='RNN', marker='v')
# 	# ax.plot(range(x_len), plot_dict['LM'], label='Levenberg–Marquardt', marker='x')
# 	ax.set_xticks(list(range(0, x_len, 2)))
# 	ax.set_xticklabels(xlabel_list[0:x_len:2], rotation=45)
# 	ax.set_title(task_name + ' Grid id: ' + str(int(grid_id)))
# 	ax.grid()
# 	ax.legend()

# 	return fig


# def plot_all_task_together(plot_dict):
# 	def get_xlabel(timestamps):
# 		xlabel_list = []
# 		for timestamp in timestamps:
# 			datetime = utility.set_time_zone(timestamp)
# 			xlabel_list.append(utility.date_time_covert_to_str(datetime))
# 		return xlabel_list

# 	def plot_by_task(axis, xlabel_list, plot_dict, task_name):
# 		x_len = len(xlabel_list)
# 		axis.plot(range(x_len), plot_dict['real'], label='Real', color='k')
# 		axis.plot(range(x_len), plot_dict['CNN_RNN'], label='CNN_RNN', linestyle='--')
# 		axis.plot(range(x_len), plot_dict['CNN_3D'], label='CNN_3D', linestyle='--')
# 		axis.plot(range(x_len), plot_dict['RNN'], label='RNN', linestyle='--')
# 		axis.set_xticks(list(range(0, x_len, 6)))
# 		axis.set_xticklabels(xlabel_list[0:x_len:6], rotation=45)

# 		axis.grid()
# 		axis.legend()
# 		axis.set_title(task_name)

# 	xlabel_list = get_xlabel(plot_dict['info'][:, 1])
# 	grid_id = plot_dict['info'][0, 0]
# 	fig = plt.figure()
# 	ax_min = fig.add_subplot(311)
# 	ax_avg = fig.add_subplot(312)
# 	ax_max = fig.add_subplot(313)

# 	min_dict = {
# 		'real': plot_dict['real'][:, 0],
# 		'CNN_RNN': plot_dict['CNN_RNN'][:, 0],
# 		'RNN': plot_dict['RNN'][:, 0],
# 		'CNN_3D': plot_dict['CNN_3D'][:, 0]
# 	}
# 	plot_by_task(ax_min, xlabel_list, min_dict, 'Task_min')

# 	avg_dict = {
# 		'real': plot_dict['real'][:, 1],
# 		'CNN_RNN': plot_dict['CNN_RNN'][:, 1],
# 		'RNN': plot_dict['RNN'][:, 1],
# 		'CNN_3D': plot_dict['CNN_3D'][:, 1]
# 	}
# 	plot_by_task(ax_avg, xlabel_list, avg_dict, 'Task_avg')

# 	max_dict = {
# 		'real': plot_dict['real'][:, 2],
# 		'CNN_RNN': plot_dict['CNN_RNN'][:, 2],
# 		'RNN': plot_dict['RNN'][:, 2],
# 		'CNN_3D': plot_dict['CNN_3D'][:, 2]
# 	}
# 	plot_by_task(ax_max, xlabel_list, max_dict, 'Task_max')
# 	plt.xlabel('time sequence')
# 	plt.ylabel('number of CDR')
# 	plt.title('Grid id:' + str(int(grid_id)))


# def plot_min_task_all_together(plot_dict):
# 	row = 0
# 	col = 0
# 	interval = (9, 40)

# 	plot_dict_min = {
# 		'info': plot_dict['info'][interval[0]:interval[1], 0, row, col],
# 		'real': plot_dict['real'][interval[0]:interval[1], 0, row, col, 0],
# 		'CNN_RNN': plot_dict['CNN_RNN'][interval[0]:interval[1], 0, row, col, 0],
# 		'CNN_3D': plot_dict['CNN_3D'][interval[0]:interval[1], 0, row, col, 0],
# 		'RNN': plot_dict['RNN'][interval[0]:interval[1], 0, row, col, 0]
# 	}

# 	grid_id = int(plot_dict_min['info'][0, 0])
# 	LM_array = get_LM(grid_id, 'min')[interval[0]:interval[1]]
# 	plot_dict_min['LM'] = LM_array
# 	# print(plot_dict_min['CNN_3D'].shape)
# 	# print(LM_array.shape)
# 	plot_all_method_together(plot_dict_min, 'Task_min')


# def plot_avg_task_all_together(plot_dict):
# 	row = 2
# 	col = 0
# 	interval = (100, 131)
# 	plot_dict_avg = {
# 		'info': plot_dict['info'][interval[0]:interval[1], 0, row, col],
# 		'real': plot_dict['real'][interval[0]:interval[1], 0, row, col, 1],
# 		'CNN_RNN': plot_dict['CNN_RNN'][interval[0]:interval[1], 0, row, col, 1],
# 		'CNN_3D': plot_dict['CNN_3D'][interval[0]:interval[1], 0, row, col, 1],
# 		'RNN': plot_dict['RNN'][interval[0]:interval[1], 0, row, col, 1]
# 	}
# 	grid_id = int(plot_dict_avg['info'][0, 0])

# 	plot_all_method_together(plot_dict_avg, 'Task_avg')


# def get_each_method():
# 	info, real, CNN_RNN_prediction = get_CNN_RNN()
# 	_, _, CNN_3D_prediction = get_CNN_3D()
# 	_, _, RNN_prediction = get_RNN()

# 	plot_dict = {
# 		'info': info,
# 		'real': real,
# 		'CNN_RNN': CNN_RNN_prediction,
# 		'CNN_3D': CNN_3D_prediction,
# 		'RNN': RNN_prediction
# 	}
# 	# row = 0
# 	# col = 0
# 	# plot_dict_1 ={
# 	# 	'info': info[:, 0, row, col],
# 	# 	'real': real[:, 0, row, col],
# 	# 	'CNN_RNN': CNN_RNN_prediction[:, 0, row, col],
# 	# 	'CNN_3D': CNN_3D_prediction[:, 0, row, col],
# 	# 	'RNN': RNN_prediction[:, 0, row, col]
# 	# }
# 	# plot_all_task_together(plot_dict_1)
# 	# plot_min_task_all_together(plot_dict)
# 	# 
# 	plot_avg_task_all_together(plot_dict)
# 	plt.show()


def evaluate_accuracy_composition():
	def task_value(obj, key_paths):
		key_paths.sort(key=itemgetter(1), reverse=False)  # sorted by task_name
		uni_task_key = []
		task_groups = []
		for k, g in groupby(key_paths, lambda x: x[1]):  # group by task
			uni_task_key.append(k)
			task_groups.append(list(g))
		task_value_dict = {}
		for task_index, each_task_name in enumerate(uni_task_key):
			each_task_group = task_groups[task_index]
			task_iter = iter(each_task_group)
			values = [reduce(getitem, ele, obj) for ele in task_iter]
			task_value_dict[each_task_name] = values

		return task_value_dict

	def plot_count_bar(data_frame, title):
		bins = np.arange(0, 1.1, 0.1, dtype=np.float)
		# print(task_max_df)
		cats_CNN_RNN_MTL = pd.cut(data_frame['CNN-RNN(MTL)'], bins)
		cats_CNN_RNN_STL = pd.cut(data_frame['CNN-RNN(STL)'], bins)
		cats_CNN_RNN_without_task = pd.cut(data_frame['CNN-RNN(*)'], bins)
		cats_CNN_3D = pd.cut(data_frame['3D CNN(MTL)'], bins)
		cats_RNN = pd.cut(data_frame['RNN(MTL)'], bins)
		cats_ARIMA = pd.cut(data_frame['ARIMA(STL)'], bins)
		cats_LM = pd.cut(data_frame['LM(STL)'], bins)
		# print(cats)
		CNN_RNN_MTL_grouped = data_frame['CNN-RNN(MTL)'].groupby(cats_CNN_RNN_MTL)
		CNN_RNN_STL_grouped = data_frame['CNN-RNN(STL)'].groupby(cats_CNN_RNN_STL)
		CNN_RNN_without_task_grouped = data_frame['CNN-RNN(*)'].groupby(cats_CNN_RNN_without_task)
		CNN_3D_grouped = data_frame['3D CNN(MTL)'].groupby(cats_CNN_3D)
		RNN_grouped = data_frame['RNN(MTL)'].groupby(cats_RNN)
		ARIMA_grouped = data_frame['ARIMA(STL)'].groupby(cats_ARIMA)
		LM_grouped = data_frame['LM(STL)'].groupby(cats_LM)

		CNN_RNN_MTL_bin_counts = CNN_RNN_MTL_grouped.count()
		CNN_RNN_STL_bin_counts = CNN_RNN_STL_grouped.count()
		CNN_RNN_without_task_bin_counts = CNN_RNN_without_task_grouped.count()
		CNN_3D_bin_counts = CNN_3D_grouped.count()
		RNN_bin_counts = RNN_grouped.count()
		ARIMA_bin_counts = ARIMA_grouped.count()
		LM_bin_counts = LM_grouped.count()
		# print(CNN_3D_bin_counts)

		bin_counts = pd.concat([ARIMA_bin_counts, LM_bin_counts, RNN_bin_counts, CNN_3D_bin_counts, CNN_RNN_without_task_bin_counts, CNN_RNN_STL_bin_counts, CNN_RNN_MTL_bin_counts], axis=1)
		bin_counts.columns = ['ARIMA(STL)', 'LM(STL)', 'RNN(MTL)', '3D CNN(MTL)', 'CNN-RNN(*)', 'CNN-RNN(STL)', 'CNN-RNN(MTL)']
		bin_counts.index = ['0~10', '10~20', '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '80~90', '90~100']
		bin_counts.index.name = 'Accuracy %'
		ax = bin_counts.plot(kind='bar', alpha=0.7, rot=0, width=0.8, figsize=(8, 4.6))
		for p in ax.patches:
			ax.annotate(str(int(p.get_height())), xy=(p.get_x(), p.get_height()))

		plt.legend(loc='upper left', fontsize='large')
		plt.title(title)
		plt.xlabel('Accuracy %', size=15)
		plt.ylabel('Number of grids', size=15)
		plt.xlim([2.5, 8.5])
		# print(bin_counts)

	def plot_KDE(data_frame, title):
		data_frame = data_frame[['ARIMA(STL)', 'LM(STL)', 'RNN(MTL)', '3D CNN(MTL)', 'CNN-RNN(*)', 'CNN-RNN(STL)', 'CNN-RNN(MTL)']]
		ax = data_frame.plot(kind='kde', title=title + ' KDE plot')
		ax.set_xlabel('Accuracy')
		ax.set_xlim(0.1, 1)
		ax.legend(loc='upper left', fontsize='large')

	def convert_to_data_frame_by_task(CNN_RNN, CNN_3D, RNN, ARIMA, CNN_RNN_STL, CNN_RNN_without_task, LM):
		CNN_RNN_key_paths = list(utility.find_in_obj(CNN_RNN, 'Accuracy'))
		CNN_3D_key_paths = list(utility.find_in_obj(CNN_3D, 'Accuracy'))
		RNN_key_paths = list(utility.find_in_obj(RNN, 'Accuracy'))
		ARIMA_key_paths = list(utility.find_in_obj(ARIMA, 'Accuracy'))
		CNN_RNN_STL_key_paths = list(utility.find_in_obj(CNN_RNN_STL, 'Accuracy'))
		CNN_RNN_without_task_key_paths = list(utility.find_in_obj(CNN_RNN_without_task, 'Accuracy'))
		LM_key_paths = list(utility.find_in_obj(LM, 'Accuracy'))

		CNN_RNN_accu_dict = task_value(CNN_RNN, CNN_RNN_key_paths)
		CNN_3D_accu_dict = task_value(CNN_3D, CNN_3D_key_paths)
		RNN_accu_dict = task_value(RNN, RNN_key_paths)
		ARIMA_accu_dict = task_value(ARIMA, ARIMA_key_paths)
		CNN_RNN_STL_accu_dict = task_value(CNN_RNN_STL, CNN_RNN_STL_key_paths)
		CNN_RNN_without_task_accu_dict = task_value(CNN_RNN_without_task, CNN_RNN_without_task_key_paths)
		LM_accu_dict = task_value(LM, LM_key_paths)
		# print(CNN_RNN_accu_dict['task_max'][-1], CNN_3D_accu_dict['task_max'][-1])

		task_max_df = pd.DataFrame({
			'CNN-RNN(MTL)': CNN_RNN_accu_dict['task_max'],
			'CNN-RNN(STL)': CNN_RNN_STL_accu_dict['task_max'],
			'CNN-RNN(*)': CNN_RNN_without_task_accu_dict['task_max'],
			'3D CNN(MTL)': CNN_3D_accu_dict['task_max'],
			'RNN(MTL)': RNN_accu_dict['task_max'],
			'LM(STL)': LM_accu_dict['task_max'],
			'ARIMA(STL)': ARIMA_accu_dict['task_max']})
		# task_max_df = task_max_df.dropna(axis=0, how='any')
		task_max_df = task_max_df.fillna(0)
		# print(task_max_df.shape)

		task_avg_df = pd.DataFrame({
			'CNN-RNN(MTL)': CNN_RNN_accu_dict['task_avg'],
			'CNN-RNN(STL)': CNN_RNN_STL_accu_dict['task_avg'],
			'CNN-RNN(*)': CNN_RNN_without_task_accu_dict['task_avg'],
			'3D CNN(MTL)': CNN_3D_accu_dict['task_avg'],
			'RNN(MTL)': RNN_accu_dict['task_avg'],
			'LM(STL)': LM_accu_dict['task_avg'],
			'ARIMA(STL)': ARIMA_accu_dict['task_avg']})
		# task_avg_df = task_avg_df.dropna(axis=0, how='any')
		task_avg_df = task_avg_df.fillna(0)

		task_min_df = pd.DataFrame({
			'CNN-RNN(MTL)': CNN_RNN_accu_dict['task_min'],
			'CNN-RNN(STL)': CNN_RNN_STL_accu_dict['task_min'],
			'CNN-RNN(*)': CNN_RNN_without_task_accu_dict['task_min'],
			'3D CNN(MTL)': CNN_3D_accu_dict['task_min'],
			'RNN(MTL)': RNN_accu_dict['task_min'],
			'LM(STL)': LM_accu_dict['task_min'],
			'ARIMA(STL)': ARIMA_accu_dict['task_min']})
		# task_min_df = task_min_df.dropna(axis=0, how='any')
		task_avg_df = task_avg_df.fillna(0)
		# task_min_df = task_min_df.fillna(0)

		return task_min_df, task_avg_df, task_max_df

	CNN_RNN_all_grid_result = './result/CNN_RNN/all_grid_result_report_0718.txt'
	CNN_3D_all_grid_result = './result/CNN_3D/all_grid_result_report.txt'
	RNN_all_grid_result = './result/RNN/all_grid_result_report.txt'
	ARIMA_all_grid_result = './result/ARIMA/all_grid_result_report_0719.txt'
	CNN_RNN_STL_all_grid_result = './result/CNN_RNN_STL/all_grid_result_report.txt'
	CNN_RNN_without_task_all_grid_result = './result/CNN_RNN_without_task/all_grid_result_report.txt'
	LM_all_grid_result = './result/LM/all_grid_result_report.txt'

	with open(CNN_RNN_all_grid_result, 'r') as fp:
		CNN_RNN = json.load(fp, encoding=None)

	with open(CNN_3D_all_grid_result, 'r') as fp:
		CNN_3D = json.load(fp, encoding=None)

	with open(RNN_all_grid_result, 'r') as fp:
		RNN = json.load(fp, encoding=None)

	with open(ARIMA_all_grid_result, 'r') as fp:
		ARIMA = json.load(fp, encoding=None)

	with open(CNN_RNN_STL_all_grid_result, 'r') as fp:
		CNN_RNN_STL = json.load(fp, encoding=None)

	with open(CNN_RNN_without_task_all_grid_result, 'r') as fp:
		CNN_RNN_without_task = json.load(fp, encoding=None)

	with open(LM_all_grid_result, 'r') as fp:
		LM = json.load(fp, encoding=None)

	task_min, task_avg, task_max = convert_to_data_frame_by_task(CNN_RNN, CNN_3D, RNN, ARIMA, CNN_RNN_STL, CNN_RNN_without_task, LM)

	print(task_min.describe())
	print(task_avg.describe())
	print(task_max.describe())

	# task_min.plot.hist()
	# task_avg.hist()
	# task_max.plot.hist()
	plot_KDE(task_min, 'Task Min')
	plot_KDE(task_avg, 'Task avg')
	plot_KDE(task_max, 'Task Max')
	# task_avg.plot(kind='kde')
	plot_count_bar(task_min, 'Task Min')
	plot_count_bar(task_avg, 'Task Avg')
	plot_count_bar(task_max, 'Task Max')

	plt.show()


def evaluate_different_method():
	def evaluate_performance(Y_real_prediction_array, file_path, divide_threshold=None):
		def print_total_report(task_report):
			for task_name, ele in task_report.items():
				print('{}: Accuracy:{:.4f} MAE:{:.4f} RMSE:{:.4f}'.format(task_name, ele['Accuracy'], ele['AE'], ele['RMSE']))

		row_center_list = list(range(40, 80, 3))
		col_center_list = list(range(30, 70, 3))
		row_range = (row_center_list[0], row_center_list[-1])
		col_range = (col_center_list[0], col_center_list[-1])
		# print((row_range[1] - row_range[0]) * (col_range[1] -  col_range[0]))
		array_len = Y_real_prediction_array.shape[0]
		if not divide_threshold:
			divide_threshold = (9 * array_len) // 10

		Y_real_prediction_array = Y_real_prediction_array[:, :, row_range[0]: row_range[1], col_range[0]: col_range[1]]
		training_data = Y_real_prediction_array[:divide_threshold]
		testing_data = Y_real_prediction_array[divide_threshold:]

		training_info = training_data[:, :, :, :, :2]
		training_real = training_data[:, :, :, :, 2:5]
		training_prediction = training_data[:, :, :, :, 5:]

		testing_info = testing_data[:, :, :, :, :2]
		testing_real = testing_data[:, :, :, :, 2:5]
		testing_prediction = testing_data[:, :, :, :, 5:]
		report_dict = report_func.report_loss_accu(testing_info, testing_real, testing_prediction, file_path)
		print_total_report(report_dict['total'])
		# print(report_dict['total'])
	
	CNN_RNN_all_grid_path = './result/CNN_RNN/all_real_prediction_traffic_array_0718.npy'
	CNN_3D_all_grid_path = './result/CNN_3D/all_real_prediction_traffic_array_0718.npy'
	RNN_all_grid_path = './result/RNN/all_real_prediction_traffic_array_0718.npy'
	ARIMA_all_grid_path = './result/ARIMA/all_real_prediction_traffic_array.npy'
	CNN_RNN_STL_all_grid_path = './result/CNN_RNN_STL/all_real_prediction_traffic_array_0715.npy'

	CNN_RNN_without_task_all_grid_path = './result/CNN_RNN_without_task/all_real_prediction_traffic_array_split_min_avg_max.npy'
	LM_all_grid_path = './result/LM/all_real_prediction_traffic_array.npy'

	# CNN_RNN_array = du.load_array(CNN_RNN_all_grid_path)
	# CNN_3D_array = du.load_array(CNN_3D_all_grid_path)
	# RNN_array = du.load_array(RNN_all_grid_path)
	# ARIMA_array = du.load_array(ARIMA_all_grid_path)
	# CNN_RNN_STL_array = du.load_array(CNN_RNN_STL_all_grid_path)
	# CNN_RNN_without_task_array = du.load_array(CNN_RNN_without_task_all_grid_path)
	LM_array = du.load_array(LM_all_grid_path)

	# evaluate_performance(CNN_RNN_array, './result/CNN_RNN/all_grid_result_report.txt')
	# evaluate_performance(CNN_3D_array, './result/CNN_3D/all_grid_result_report.txt')
	# evaluate_performance(RNN_array, './result/RNN/all_grid_result_report.txt')
	# evaluate_performance(ARIMA_array, './result/ARIMA/all_grid_result_report.txt')
	# evaluate_performance(CNN_RNN_STL_array, './result/CNN_RNN_STL/all_grid_result_report.txt')
	# evaluate_performance(CNN_RNN_without_task_array, './result/CNN_RNN_without_task/all_grid_result_report.txt')
	evaluate_performance(LM_array, './result/LM/all_grid_result_report.txt', 0)

def evaluate_MTL_and_STL():
	def task_value(obj, key_paths):
		key_paths.sort(key=itemgetter(1), reverse=False)  # sorted by task_name
		uni_task_key = []
		task_groups = []
		for k, g in groupby(key_paths, lambda x: x[1]):  # group by task
			uni_task_key.append(k)
			task_groups.append(list(g))
		task_value_dict = {}
		for task_index, each_task_name in enumerate(uni_task_key):
			each_task_group = task_groups[task_index]
			task_iter = iter(each_task_group)
			# values = [reduce(getitem, ele, obj) for ele in task_iter]
			grid_id_accu_list = []
			for ele in task_iter:
				accu = reduce(getitem, ele, obj)
				grid_id_accu_list.append((ele[0], accu))  # (grid_id, accu)
				grid_id_accu_list = sorted(grid_id_accu_list, key=itemgetter(0))  # sort by grid id
			task_value_dict[each_task_name] = grid_id_accu_list

		return task_value_dict

	def convert_to_data_frame_by_task(method_dict):

		def get_data_frame(method_dict, task_name):
			MTL_task_df = pd.DataFrame({
				'Grid_id': [ele[0] for ele in method_dict['CNN_RNN_MTL'][task_name]],
				'MTL': [ele[1] for ele in method_dict['CNN_RNN_MTL'][task_name]]})
			MTL_task_df = MTL_task_df.set_index('Grid_id')

			STL_task_df = pd.DataFrame({
				'Grid_id': [ele[0] for ele in method_dict['CNN_RNN_STL'][task_name]],
				'STL': [ele[1] for ele in method_dict['CNN_RNN_STL'][task_name]]})
			STL_task_df = STL_task_df.set_index('Grid_id')
			task_df = pd.concat([MTL_task_df, STL_task_df], axis=1, join='outer')
			task_df = task_df.dropna(axis=0, how='any')
			return task_df

		for key, obj in method_dict.items():
			obj_key_path = list(utility.find_in_obj(obj, 'Accuracy'))  # key_path :[grid_id, task_type, 'Accuracy']
			acc_dict = task_value(obj, obj_key_path)
			method_dict[key] = acc_dict

		max_df = get_data_frame(method_dict, 'task_max')
		avg_df = get_data_frame(method_dict, 'task_avg')
		min_df = get_data_frame(method_dict, 'task_min')
		return min_df, avg_df, max_df
		# print(method_dict)

	def plot_improvement_heat_map(task_df):
		improve_df = (task_df.loc[:, 'MTL'] - task_df.loc[:, 'STL']) * 100
		improve_df.drop('total', inplace=True)
		data_array = np.zeros([100, 100], dtype=float)
		for i, value in enumerate(improve_df.values):
			grid_id = int(improve_df.index[i])
			row, col = utility.compute_row_col(grid_id)
			data_array[row, col] = value

		plt.imshow(data_array.T, vmin=-10, vmax=10, cmap=plt.get_cmap('bwr'))
		plt.grid(True)
		plt.colorbar()
		plt.show()

	def compare_two_task(task_df):
		task_df['larger'] = task_df.apply(lambda x: 'MTL' if x['MTL'] > x['STL'] else 'STL', axis=1)
		print(task_df.describe())
		df_larger = task_df.loc[task_df['larger'] == 'MTL']
		# df_larger['impove'] = task_df.apply(lambda x: (x['MTL'] - x['STL']), axis=1)
		df_improve = (df_larger.loc[:, 'MTL'] - df_larger.loc[:, 'STL']) * 100
		print(df_improve.describe())
		# print(task_df['larger'].value_counts())

	CNN_RNN_MTL_all_grid_result = './result/CNN_RNN/all_grid_result_report_0718.txt'
	CNN_RNN_STL_all_grid_result = './result/CNN_RNN_STL/all_grid_result_report.txt'

	with open(CNN_RNN_MTL_all_grid_result, 'r') as fp:
		CNN_RNN_MTL = json.load(fp, encoding=None)

	with open(CNN_RNN_STL_all_grid_result, 'r') as fp:
		CNN_RNN_STL = json.load(fp, encoding=None)

	method_dict = {
		'CNN_RNN_MTL': CNN_RNN_MTL,
		'CNN_RNN_STL': CNN_RNN_STL
	}
	min_task, avg_task, max_task = convert_to_data_frame_by_task(method_dict)
	# print(max_task.idxmax())
	# compare_two_task(min_task)
	# compare_two_task(avg_task)
	# compare_two_task(max_task)
	plot_improvement_heat_map(max_task)


def evaluate_CNN_RNN_without_task():
	def search_grid(data_array, grid_id):
		array = np.transpose(data_array, (2, 3, 0, 1, 4))
		for row in range(array.shape[0]):
			for col in range(array.shape[1]):
				if grid_id == array[row, col, 0, 0, 0]:
					return row, col
		return 0, 0

	def get_data():
		method_result_path = os.path.join(root_dir, 'CNN_RNN/result/CNN_RNN_without_task/all_real_prediction_traffic_array.npy')
		result_array = du.load_array(method_result_path)

		row_center_list = list(range(40, 80, 3))
		col_center_list = list(range(30, 70, 3))
		row_range = range(row_center_list[0] - 1, row_center_list[-1] + 1)
		col_range = range(col_center_list[0] - 1, col_center_list[-1] + 1)
		logger.info('row_range {}:{} col_range: {}:{}'.format(row_range[0], row_range[-1], col_range[0], col_range[-1]))
		result_array = result_array[:-1, :, :row_range[-1] - row_range[0] + 1, :col_range[-1] - col_range[0] + 1]
		logger.debug('result_array shape:{}'.format(result_array.shape))
		return result_array

	def evaluate_performance(real, prediction):
		# data_array_len = real.shape[0]

		# test_real = real[9 * data_array_len // 10:]
		# test_prediction = prediction[9 * data_array_len // 10:]

		MAPE_loss = utility.MAPE_loss(real, prediction)
		AE_loss = utility.AE_loss(real, prediction)
		RMSE_loss = utility.RMSE_loss(real, prediction)
		# MAPE_train = utility.MAPE_loss(train_array[:, :, :, :, 2, np.newaxis], train_array[:, :, :, :, 3, np.newaxis])
		# print('test accu:{} test AE:{} test RMSE:{}'.format(1 - MAPE_test, AE_test, RMSE_test))
		return 1 - MAPE_loss, AE_loss, RMSE_loss

	def calculate_min_avg_max(data_array):
		new_data_array = np.zeros((data_array.shape[0], 1, 100, 100, 8))  # hour, 1, row, col, (grid_id, timestmap, real_min, real_avg, real_max, preidiction_min, prediction_avg, prediction_max)
		data_array = np.transpose(data_array, (0, 2, 3, 1, 4))  # hour, row, col, 10min, feature
		for i in range(data_array.shape[0]):
			for row in range(data_array.shape[1]):
				for col in range(data_array.shape[2]):
					real_max_value = np.amax(data_array[i, row, col, :, 2])
					prediction_max_value = np.amax(data_array[i, row, col, :, 3])

					real_min_value = np.amin(data_array[i, row, col, :, 2])
					prediction_min_value = np.amin(data_array[i, row, col, :, 3])

					real_avg_value = np.mean(data_array[i, row, col, :, 2])
					prediction_avg_value = np.mean(data_array[i, row, col, :, 3])

					grid_id = data_array[i, row, col, 0, 0]
					timestamp = data_array[i, row, col, 0, 1]
					row_index, col_index = utility.compute_row_col(grid_id)
					new_data_array[i, 0, row_index, col_index, 0] = grid_id
					new_data_array[i, 0, row_index, col_index, 1] = timestamp
					new_data_array[i, 0, row_index, col_index, 2] = real_min_value
					new_data_array[i, 0, row_index, col_index, 3] = real_avg_value
					new_data_array[i, 0, row_index, col_index, 4] = real_max_value
					new_data_array[i, 0, row_index, col_index, 5] = prediction_min_value
					new_data_array[i, 0, row_index, col_index, 6] = prediction_avg_value
					new_data_array[i, 0, row_index, col_index, 7] = prediction_max_value
					# logger.info('grid_id:{} real:{} prediction:{}'.format(int(grid_id), real_max_value, prediction_max_value))

		return new_data_array

	def plot_CNN_RNN_without_task(data_arrray, grid_id, interval=6):
		logger.debug('data_arrray :{}'.format(data_arrray.shape))
		# plot_row = 10
		# plot_col = 30
		plot_row, plot_col = search_grid(data_arrray, grid_id)
		# result_array_len = result_array.shape[0]
		logger.info('plot_row:{} plot_col:{}'.format(plot_row, plot_col))
		plot_real = data_arrray[:, :, plot_row, plot_col, 2].reshape(-1, 1)
		plot_prediction = data_arrray[:, :, plot_row, plot_col, 3].reshape(-1, 1)
		plt_info = data_arrray[:, :, plot_row, plot_col, :2].reshape(-1, 2)
		report_func.plot_predict_vs_real(plt_info, plot_real, plot_prediction, 'CNN-RNN(*) prediction on ', interval)

	def evaluate_one_grid(origin_array, real_preidction, grid_id=4867):
		logger.info('origin_array shape:{} real_preidction shape:{}'.format(origin_array.shape, real_preidction.shape))
		plot_CNN_RNN_without_task(origin_array[-149:], grid_id, 24)
		row, col = search_grid(real_preidction, grid_id)

		accu_min, AE_min, RMSE_min = evaluate_performance(real_preidction[-149:, :, row: row + 1, col: col + 1, 2], real_preidction[-149:, :, row: row + 1, col: col + 1, 5])
		accu_avg, AE_avg, RMSE_avg = evaluate_performance(real_preidction[-149:, :, row: row + 1, col: col + 1, 3], real_preidction[-149:, :, row: row + 1, col: col + 1, 6])
		accu_max, AE_max, RMSE_max = evaluate_performance(real_preidction[-149:, :, row: row + 1, col: col + 1, 4], real_preidction[-149:, :, row: row + 1, col: col + 1, 7])
		logger.info('grid id:{} MIN accu:{} AE:{} RMSE:{}'.format(grid_id, accu_min, AE_min, RMSE_min))
		logger.info('grid id:{} AVG accu:{} AE:{} RMSE:{}'.format(grid_id, accu_avg, AE_avg, RMSE_avg))
		logger.info('grid id:{} MAX accu:{} AE:{} RMSE:{}'.format(grid_id, accu_max, AE_max, RMSE_max))

		plot_CNN_RNN_without_task(real_preidction[-149:, :, :, :, (0, 1, 4, 7)], grid_id, 2)

	reload = None
	result_array = get_data()
	accu, AE, RMSE = evaluate_performance(result_array[-149:, :, :, :, 2], result_array[-149:, :, :, :, 3])
	logger.info('total data: test accu:{} test AE:{} test RMSE:{}'.format(accu, AE, RMSE))
	if reload:
		real_preidction = calculate_min_avg_max(result_array)
		du.save_array(real_preidction, os.path.join(root_dir, 'CNN_RNN/result/CNN_RNN_without_task/all_real_prediction_traffic_array_split_min_avg_max.npy'))
	else:
		real_preidction = du.load_array(os.path.join(root_dir, 'CNN_RNN/result/CNN_RNN_without_task/all_real_prediction_traffic_array_split_min_avg_max.npy'))
	print()
	accu_min, AE_min, RMSE_min = evaluate_performance(real_preidction[-149:, :, :, :, 2], real_preidction[-149:, :, :, :, 5])
	accu_avg, AE_avg, RMSE_avg = evaluate_performance(real_preidction[-149:, :, :, :, 3], real_preidction[-149:, :, :, :, 6])
	accu_max, AE_max, RMSE_max = evaluate_performance(real_preidction[-149:, :, :, :, 4], real_preidction[-149:, :, :, :, 7])
	logger.info('MIN accu:{} AE:{} RMSE:{}'.format(accu_min, AE_min, RMSE_min))
	logger.info('AVG accu:{} AE:{} RMSE:{}'.format(accu_avg, AE_avg, RMSE_avg))
	logger.info('MAX accu:{} AE:{} RMSE:{}'.format(accu_max, AE_max, RMSE_max))

	evaluate_one_grid(result_array, real_preidction, 4867)
	plt.show()


def evaluate_MTL_and_without_task():
	def plot_method_together_1(plot_dict, title_name, interval=6):

		def plot_by_axis(axis, xlabel_list, plot_dict):
			method_name = plot_dict['name']
			x_len = len(xlabel_list)
			axis.plot(range(x_len), plot_dict['real'], label='Real', color='k')
			axis.plot(range(x_len), plot_dict['prediction'], label='Prediction', color='r', linestyle='--')
			axis.set_xticks(list(range(0, x_len, interval)))
			axis.set_xticklabels(xlabel_list[0:x_len:interval], rotation=45)

			axis.grid()
			axis.legend()
			axis.set_title(method_name)

		def get_xlabel(timestamps):
			xlabel_list = []
			for timestamp in timestamps:
				datetime = utility.set_time_zone(timestamp)
				xlabel_list.append(utility.date_time_covert_to_str(datetime))
			return xlabel_list

		fig, ax = plt.subplots(2, 1, figsize=(10, 4))
		logger.debug('ax shape:{}'.format(ax.shape))
		index = 0
		for method_key, method in plot_dict.items():
			xlabel_list = get_xlabel(method['info'][:, 1])
			plot_by_axis(ax[index], xlabel_list, method)
			index += 1

	def plot_method_together_2(plot_dict, title_name, interval=6):
		def get_xlabel(timestamps):
			xlabel_list = []
			for timestamp in timestamps:
				datetime = utility.set_time_zone(timestamp)
				xlabel_list.append(utility.date_time_covert_to_str(datetime))
			return xlabel_list

		fig, ax = plt.subplots(1, 1, figsize=(10, 4))
		key_list = list(plot_dict.keys())
		xlabel_list = get_xlabel(plot_dict[key_list[0]]['info'][:, 1])
		x_len = len(xlabel_list)
		real = plot_dict[key_list[0]]['real']
		grid_id = plot_dict[key_list[0]]['info'][0, 0]
		logger.debug('grid_id:{} xlabel_list len:{} real shape:{}'.format(grid_id, x_len, real.shape))
		ax.plot(range(x_len), real, label='Real', color='k')

		for method_key, method_dict in plot_dict.items():
			ax.plot(range(x_len), method_dict['prediction'], label=str(method_key), color=method_dict['color'], linestyle='--')

		ax.set_xticks(list(range(0, x_len, interval)))
		ax.set_xticklabels(xlabel_list[0:x_len:interval], rotation=45)
		ax.set_title(title_name + ' grid id ' + str(int(grid_id)))
		ax.set_xlabel('Times')
		ax.set_ylabel('Number of CDRs')
		ax.grid()
		ax.legend()

	def plot_together(MTL_array, without_task_array):
		plt_dict = {}
		grid_id = 4867
		row_index, col_index = utility.compute_row_col(grid_id)
		logger.info('row_index:{} col_index:{}'.format(row_index, col_index))

		plot_MTL_real_min = MTL_array[-149:, :, row_index, col_index, 2].reshape(-1, 1)  # min
		plot_MTL_real_avg = MTL_array[-149:, :, row_index, col_index, 3].reshape(-1, 1)  # avg
		plot_MTL_real_max = MTL_array[-149:, :, row_index, col_index, 4].reshape(-1, 1)  # max


		plot_MTL_predinction_min = MTL_array[-149:, :, row_index, col_index, 5].reshape(-1, 1)  # min
		plot_MTL_predinction_avg = MTL_array[-149:, :, row_index, col_index, 6].reshape(-1, 1)  # avg
		plot_MTL_predinction_max = MTL_array[-149:, :, row_index, col_index, 7].reshape(-1, 1)  # max

		plot_without_task_array_real_min = without_task_array[-149:, :, row_index, col_index, 2].reshape(-1, 1)  # min
		plot_without_task_array_real_avg = without_task_array[-149:, :, row_index, col_index, 3].reshape(-1, 1)  # avg
		plot_without_task_array_real_max = without_task_array[-149:, :, row_index, col_index, 4].reshape(-1, 1)  # max


		plot_without_task_array_predinction_min = without_task_array[-149:, :, row_index, col_index, 5].reshape(-1, 1)  # min
		plot_without_task_array_predinction_avg = without_task_array[-149:, :, row_index, col_index, 6].reshape(-1, 1)  # avg
		plot_without_task_array_predinction_max = without_task_array[-149:, :, row_index, col_index, 7].reshape(-1, 1)  # max

		plt_MTL_dict = {
			'name': 'CNN-RNN(MTL)',
			'real': plot_MTL_real_max,
			'prediction': plot_MTL_predinction_max,
			'info': MTL_array[-149:, :, row_index, col_index, :2].reshape(-1, 2),
			'color': 'r'
		}

		plt_without_task_dict = {
			'name': 'CNN-RNN(*)',
			'real': plot_without_task_array_real_max,
			'prediction': plot_without_task_array_predinction_max,
			'info': without_task_array[-149:, :, row_index, col_index, :2].reshape(-1, 2),
			'color': 'dimgray'
		}
		plt_dict['CNN-RNN(MTL)'] = plt_MTL_dict
		plt_dict['CNN-RNN(*)'] = plt_without_task_dict

		plot_method_together_2(plt_dict, 'Max traffic prediction on')
		plt.show()

	def get_data():
		CNN_RNN_all_grid_path = './result/CNN_RNN/all_real_prediction_traffic_array_0718.npy'
		CNN_RNN_without_task_all_grid_path = './result/CNN_RNN_without_task/all_real_prediction_traffic_array_split_min_avg_max.npy'
		CNN_RNN_MTL_array = du.load_array(CNN_RNN_all_grid_path)
		CNN_RNN_without_task_array = du.load_array(CNN_RNN_without_task_all_grid_path)
		CNN_RNN_MTL_array = CNN_RNN_MTL_array[:-1]
		logger.info('CNN_RNN_MTL_array shape:{} CNN_RNN_without_task_array shape:{}'.format(CNN_RNN_MTL_array.shape, CNN_RNN_without_task_array.shape))
		return CNN_RNN_MTL_array, CNN_RNN_without_task_array

	CNN_RNN_MTL_array, CNN_RNN_without_task_array = get_data()
	plot_together(CNN_RNN_MTL_array, CNN_RNN_without_task_array)


if __name__ == '__main__':
	# evaluate_MTL_and_STL()
	# evaluate_different_method()
	# evaluate_accuracy_composition()
	# plot_method()
	evaluate_CNN_RNN_without_task()
	# evaluate_MTL_and_without_task()