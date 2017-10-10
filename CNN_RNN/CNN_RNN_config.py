import os
import json
import numpy as np
# from CNN_RNN import CNN_RNN_2
from collections import OrderedDict
import operator
import itertools
from functools import reduce
import shutil
import random
from pprint import pprint


class HyperParameterConfig:
	def __init__(self):
		self.iter_epoch = 2000
		self.batch_size = 100
		self.learning_rate = 0.0014
		self.keep_rate = 0.85
		self.weight_decay = 0.001

		'''share layer Dense'''
		self.fully_connected_W_init = 'xavier'
		self.fully_connected_units = 512

		'''prediction layer'''
		self.prediction_layer_1_W_init = 'xavier'
		self.prediction_layer_1_uints = 64

		self.prediction_layer_2_W_init = 'xavier'
		self.prediction_layer_keep_rate = 0.85

	def CNN_RNN(self):
		'''build_bi_RNN_network'''
		self.RNN_num_layers = 3
		self.RNN_num_step = 6
		self.RNN_hidden_node_size = 256
		self.RNN_cell = 'LSTMcell'
		self.RNN_cell_init_args = {'forget_bias': 1.0, 'use_peepholes': True, 'state_is_tuple': True}
		self.RNN_init_state_noise_stddev = 0.2
		self.RNN_initializer = 'xavier'

		'''build_CNN_network'''
		self.CNN_layer_activation_fn = 'relu'
		self.CNN_layer_1_5x5_kernel_shape = [5, 5, 2, 32]
		self.CNN_layer_1_5x5_kernel_strides = [1, 1, 1, 1]
		self.CNN_layer_1_5x5_conv_Winit = 'xavier'

		self.CNN_layer_1_5x5_pooling = 'max_pool'
		self.CNN_layer_1_5x5_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_1_5x5_pooling_strides = [1, 2, 2, 1]

		self.CNN_layer_1_3x3_kernel_shape = [3, 3, 2, 32]
		self.CNN_layer_1_3x3_kernel_strides = [1, 1, 1, 1]
		self.CNN_layer_1_3x3_conv_Winit = 'xavier'

		self.CNN_layer_1_3x3_pooling = 'max_pool'
		self.CNN_layer_1_3x3_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_1_3x3_pooling_strides = [1, 2, 2, 1]

		self.CNN_layer_1_pooling = 'avg_pool'
		self.CNN_layer_1_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_1_pooling_strides = [1, 2, 2, 1]

		self.CNN_layer_2_kernel_shape = [5, 5, 64, 64]
		self.CNN_layer_2_strides = [1, 1, 1, 1]
		self.CNN_layer_2_conv_Winit = 'xavier'

		self.CNN_layer_2_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_2_pooling_strides = [1, 1, 1, 1]
		self.CNN_layer_2_pooling = 'avg_pool'


	def CNN_RNN_2(self):
		'''build_bi_RNN_network'''
		self.CNN_layer_activation_fn = 'relu'
		self.RNN_num_layers = 2
		self.RNN_num_step = 6
		self.RNN_hidden_node_size = 128
		self.RNN_cell = 'LSTMcell'
		self.RNN_cell_init_args = {'forget_bias': 1.0, 'use_peepholes': True, 'state_is_tuple': True}
		self.RNN_init_state_noise_stddev = 0.2
		self.RNN_initializer = 'xavier'

		'''build_CNN_network'''
		self.CNN_layer_1_kernel_shape = [5, 5, 1, 32]
		self.CNN_layer_1_strides = [1, 1, 1, 1]
		self.CNN_layer_1_conv_Winit = 'xavier'

		self.CNN_layer_1_pooling = 'max_pool'
		self.CNN_layer_1_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_1_pooling_strides = [1, 2, 2, 1]

		self.CNN_layer_2_kernel_shape = [5, 5, 32, 64]
		self.CNN_layer_2_strides = [1, 1, 1, 1]
		self.CNN_layer_2_conv_Winit = 'xavier'

		self.CNN_layer_2_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_2_pooling_strides = [1, 1, 1, 1]
		self.CNN_layer_2_pooling = 'avg_pool'

		self.CNN_layer_3_kernel_shape = [5, 5, 64, 64]
		self.CNN_layer_3_strides = [1, 1, 1, 1]
		self.CNN_layer_3_conv_Winit = 'xavier'

		self.CNN_layer_3_pooling_ksize = [1, 2, 2, 1]
		self.CNN_layer_3_pooling_strides = [1, 1, 1, 1]
		self.CNN_layer_3_pooling = 'avg_pool'

	def get_variable(self):
		# total = vars(self)
		total = self.__dict__
		key_var = {}
		for key, value in total.items():
			if key.startswith('__') or callable(value):
				continue
			key_var[key] = value
		# print(key_var)
		return key_var

	def get_json_str(self):
		key_var = self.get_variable()
		json_string = json.dumps(key_var, sort_keys=True, indent=4)
		return json_string

	def save_json(self, file_path='./result/temp.json'):
		key_var = self.get_variable()
		if not os.path.exists(os.path.dirname(file_path)):
			os.makedirs(os.path.dirname(file_path))

		with open(file_path, 'w') as outfile:
			json.dump(key_var, outfile, sort_keys=True, indent=4)

	def read_config(self, file_path='./result/temp.json'):
		with open(file_path, 'r') as data_file:
			data = json.load(data_file)
		for key, value in data.items():
			setattr(self, key, value)


class GridSearch():
	def __init__(self, X_array, Y_array):
		self.X_array = X_array
		self.Y_array = Y_array
		# model_basename = os.path.basename(model_path)
		# self.model_dirname = os.path.dirname(model_path)
		# self.model_name = os.path.splitext(model_basename)
		self.hyper_config = None
		self.basic_result_path = ''
		self.search_task_name = ''

	def _find_in_obj(self, obj, condition, path=None):
		if path is None:
			path = []

		if isinstance(obj, list):
			for index, value in enumerate(obj):
				new_path = list(path)
				new_path.append(index)
				for result in self._find_in_obj(value, condition, path=new_path):
					yield result
		if isinstance(obj, dict):
			for key, value in obj.items():
				new_path = list(path)
				new_path.append(key)
				for result in self._find_in_obj(value, condition, path=new_path):
					yield result

				if condition == key:
					new_path = list(path)
					new_path.append(key)
					yield new_path

	def _find_highest_accu(self, obj):
			key_paths = self._find_in_obj(obj, 'testing_accurcy')  # generator

			key_paths = list(key_paths)
			key_paths.sort(key=operator.itemgetter(1))  # sort by element 2 in tuple
			uni_task_key = []
			task_groups = []
			max_pairs = []
			for k, g in itertools.groupby(key_paths, lambda x: x[1]):  # group by task
				uni_task_key.append(k)
				task_groups.append(list(g))
				# value = reduce(operator.getitem, key_pair, obj)
			for each_task_group in task_groups:
				task_iter = iter(each_task_group)
				key_value = [(ele, reduce(operator.getitem, ele, obj)) for ele in task_iter]
				max_pair = max(key_value, key=lambda x: x[1])

				'''
				for ele in task_iter:
					value = reduce(operator.getitem, ele, obj)
					print('rate:{} task:{} {}:{}'.format(
						ele[0],
						ele[1],
						ele[2],
						value))
				'''
				max_pairs.append(max_pair)
			return max_pairs

	def search_learning_rate(self):
		hyper_config = HyperParameterConfig()
		rate_summerize = OrderedDict()
		# rate_list = list(range(0.0001, 0.05, 0.0005))
		rate_array = np.arange(0.001, 0.004, 0.0005)
		rate_array = rate_array.tolist()

		# basic_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/search_learning_rate/'
		basic_result_path = './result/search_learning_rate/'
		# shutil.rmtree(basic_result_path)
		if not os.path.exists(basic_result_path):
			os.makedirs(basic_result_path)

		for rate in rate_array:
			print('rate:{}'.format(rate))
			hyper_config.learning_rate = round(float(rate), 6)
			save_model_path = self.model_dirname + '/' + self.model_name[0] + '_rate_' + str(hyper_config.learning_rate) + '.ckpt'
			result_path = basic_result_path + 'rate_' + str(hyper_config.learning_rate) + '/'

			if not os.path.exists(result_path):
				os.makedirs(result_path)

			model_path = {
				'reload_path': './output_model/CNN_RNN_test.ckpt',
				'save_path': save_model_path,
				'result_path': result_path
			}
			summerize = self._run_CNN_RNN(model_path, hyper_config)
			rate_summerize[str(rate)] = summerize

		print('search finish!!')

		max_pairs = self._find_highest_accu(rate_summerize)
		for max_pair in max_pairs:
			print('rate:{} in task: {} {}:{}'.format(
				max_pair[0][0],
				max_pair[0][1],
				max_pair[0][2],
				max_pair[1]))

		with open(basic_result_path + 'search_rate_report.txt', 'w') as outfile:
			json_string = json.dumps(rate_summerize, sort_keys=False, indent=4)
			# print(json_string)
			outfile.write(json_string + '\n')
			for max_pair in max_pairs:
				outfile.write('rate:{} in task: {} {}:{} \n'.format(
					max_pair[0][0],
					max_pair[0][1],
					max_pair[0][2],
					max_pair[1]))

		'''
		with open(basic_result_path + 'search_rate_report.txt', 'w') as outfile:
			for rate_v, rate_v_element in rate_summerize.items():
				print(rate_v, ': ')
				outfile.write()
				for task, result in rate_v_element.items():
					print(task, ':')
					print('\t', result)
		'''

	def search_keep_rate(self):
		hyper_config = HyperParameterConfig()
		summerized = OrderedDict()

		keep_rate_array = np.arange(0.3, 1., 0.05)
		keep_rate_array = keep_rate_array.tolist()
		# basic_result_path = '/home/mldp/ML_with_bigdata/CNN_RNN/result/search_keep_rate/'
		basic_result_path = './result/search_keep_rate/'

		if not os.path.exists(basic_result_path):
			os.makedirs(basic_result_path)

		for keep_rate in keep_rate_array:
			print('keep_rate:{}'.format(keep_rate))
			hyper_config.keep_rate = round(float(keep_rate), 6)
			save_model_path = self.model_dirname + '/' + self.model_name[0] + '_keep_rate_' + str(hyper_config.keep_rate) + '.ckpt'
			result_path = basic_result_path + 'keep_rate_' + str(hyper_config.keep_rate) + '/'

			if not os.path.exists(result_path):
				os.makedirs(result_path)

			model_path = {
				'reload_path': './output_model/CNN_RNN_test.ckpt',
				'save_path': save_model_path,
				'result_path': result_path
			}
			summerize = self._run_CNN_RNN(model_path, hyper_config)
			summerized[str(keep_rate)] = summerize

		print('search finish!!')
		max_pairs = self._find_highest_accu(summerized)
		for max_pair in max_pairs:
			print('rate:{} in task: {} {}:{}'.format(
				max_pair[0][0],
				max_pair[0][1],
				max_pair[0][2],
				max_pair[1]))

		with open(basic_result_path + 'search_keep_rate_report.txt', 'w') as outfile:
			json_string = json.dumps(summerized, sort_keys=False, indent=4)
			# print(json_string)
			outfile.write(json_string + '\n')
			for max_pair in max_pairs:
				outfile.write('keep_rate:{} in task: {} {}:{} \n'.format(
					max_pair[0][0],
					max_pair[0][1],
					max_pair[0][2],
					max_pair[1]))

	def search_predition_keep_rate(self):
		self.hyper_config = HyperParameterConfig()
		set_attr = 'prediction_layer_keep_rate'
		iterator = np.arange(0.3, 1., 0.05)
		self.search_task_name = 'serach_precition_keep_rate'
		self.basic_result_path = os.path.join('./result')
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)

		self._run_grid_search(iterator, set_attr)

	def search_weight_decay(self):
		self.hyper_config = HyperParameterConfig()
		set_attr = 'weight_decay'
		iterator = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
		self.search_task_name = 'search_weight_decay'
		self.basic_result_path = os.path.join('./result')
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)
		self._run_grid_search(iterator, set_attr)

	def search_batch_size(self):
		self.hyper_config = HyperParameterConfig()
		set_attr = 'batch_size'
		iterator = range(10, 130, 10)
		self.search_task_name = 'search_batch_size'
		self.basic_result_path = os.path.join('./result')
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)
		self._run_grid_search(iterator, set_attr)

	def search_RNN_layer(self):
		self.hyper_config = HyperParameterConfig()
		set_attr = 'RNN_num_layers'
		iterator = range(2, 7, 1)
		self.search_task_name = 'search_RNN_layer'
		self.basic_result_path = os.path.join('./result')
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)
		self._run_grid_search(iterator, set_attr)

	def search_RNN_nodes(self):
		self.hyper_config = HyperParameterConfig()
		set_attr = 'RNN_hidden_node_size'
		index = range(4, 11)
		iterator = map(lambda x: 2 ** x, index)
		self.search_task_name = 'search_RNN_nodes'
		self.basic_result_path = os.path.join('./result')
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)
		self._run_grid_search(iterator, set_attr)

	def search_prediction_nodes(self):
		self.hyper_config = HyperParameterConfig()
		set_attr = 'prediction_layer_1_uints'
		index = range(3, 11)
		iterator = map(lambda x: 2 ** x, index)
		self.search_task_name = 'search_prediction_nodes'
		self.basic_result_path = os.path.join('./result')
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)
		self._run_grid_search(iterator, set_attr)

	def search_fully_connected_nodes(self):
		self.hyper_config = HyperParameterConfig()
		set_attr = 'fully_connected_units'
		index = range(3, 11)
		iterator = map(lambda x: 2 ** x, index)
		self.search_task_name = 'search_fully_connected_nodes'
		self.basic_result_path = os.path.join('./result')
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)
		self._run_grid_search(iterator, set_attr)

	def _search_CNN_RNN(self):
		def random_choise(hyper_list):
			return random.choice(hyper_list)

		base_index = range(4, 7)
		kernal_size_list = list(map(lambda x: 2 ** x, base_index))
		layer_1_kernel_size = random_choise(kernal_size_list)
		layer_2_kernel_size = random_choise(kernal_size_list)

		self.hyper_config.CNN_layer_1_5x5_kernel_shape[3] = layer_1_kernel_size
		self.hyper_config.CNN_layer_1_3x3_kernel_shape[3] = layer_1_kernel_size
		self.hyper_config.CNN_layer_2_kernel_shape[2] = layer_1_kernel_size * 2
		self.hyper_config.CNN_layer_2_kernel_shape[3] = layer_2_kernel_size

		pooling_list = ['max_pool', 'avg_pool']
		pooling_1 = random_choise(pooling_list)
		self.hyper_config.CNN_layer_1_5x5_pooling = pooling_1
		self.hyper_config.CNN_layer_1_5x5_pooling = pooling_1
		self.hyper_config.CNN_layer_1_pooling = random_choise(pooling_list)
		self.hyper_config.CNN_layer_2_pooling = random_choise(pooling_list)

	def _search_CNN_RNN_2(self):
		def random_choise(hyper_list):
			return random.choice(hyper_list)

		pooling_list = ['max_pool', 'avg_pool']
		ksize_list = [[1, 2, 2, 1], [1, 3, 3, 1], [1, 4, 4, 1], [1, 5, 5, 1], [1, 6, 6, 1]]
		stride_list = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 3, 3, 1], [1, 5, 5, 1], [1, 6, 6, 1]]

		base_index = range(4, 7)
		kernal_size_list = list(map(lambda x: 2 ** x, base_index))
		kernel_height_width_list = list(range(2, 7))

		layer_1_kernel_size = random_choise(kernal_size_list)
		layer_2_kernel_size = random_choise(kernal_size_list)
		layer_3_kernel_size = random_choise(kernal_size_list)

		layer_1_kernel_height_width = random_choise(kernel_height_width_list)
		layer_2_kernel_height_width = random_choise(kernel_height_width_list)
		layer_3_kernel_height_width = random_choise(kernel_height_width_list)

		# self.hyper_config.CNN_layer_1_kernel_shape[3] = layer_1_kernel_size
		self.hyper_config.CNN_layer_1_kernel_shape[0] = layer_1_kernel_height_width
		self.hyper_config.CNN_layer_1_kernel_shape[1] = layer_1_kernel_height_width

		self.hyper_config.CNN_layer_1_kernel_shape[0] = layer_2_kernel_height_width
		self.hyper_config.CNN_layer_1_kernel_shape[1] = layer_2_kernel_height_width
		# self.hyper_config.CNN_layer_2_kernel_shape[2] = layer_1_kernel_size
		# self.hyper_config.CNN_layer_2_kernel_shape[3] = layer_2_kernel_size

		self.hyper_config.CNN_layer_3_kernel_shape[0] = layer_3_kernel_height_width
		self.hyper_config.CNN_layer_3_kernel_shape[1] = layer_3_kernel_height_width
		# self.hyper_config.CNN_layer_3_kernel_shape[2] = layer_2_kernel_size
		# self.hyper_config.CNN_layer_3_kernel_shape[3] = layer_3_kernel_size

		self.hyper_config.CNN_layer_1_strides = random_choise(stride_list)
		self.hyper_config.CNN_layer_2_strides = random_choise(stride_list)
		self.hyper_config.CNN_layer_3_strides = random_choise(stride_list)

		self.hyper_config.CNN_layer_1_pooling = random_choise(pooling_list)
		self.hyper_config.CNN_layer_2_pooling = random_choise(pooling_list)
		self.hyper_config.CNN_layer_3_pooling = random_choise(pooling_list)

		self.hyper_config.CNN_layer_1_pooling_ksize = random_choise(ksize_list)
		self.hyper_config.CNN_layer_2_pooling_ksize = random_choise(ksize_list)
		self.hyper_config.CNN_layer_3_pooling_ksize = random_choise(ksize_list)

		self.hyper_config.CNN_layer_1_pooling_strides = random_choise(stride_list)
		self.hyper_config.CNN_layer_2_pooling_strides = random_choise(stride_list)
		self.hyper_config.CNN_layer_3_pooling_strides = random_choise(stride_list)

	def random_grid_search(self, task_name='random_search'):
		base_index = range(5, 11)
		fully_connected_list = list(map(lambda x: 2 ** x, base_index))
		prediction_nodes_list = list(map(lambda x: 2 ** x, base_index))
		RNN_nodes_list = list(map(lambda x: 2 ** x, base_index))
		RNN_layer_list = list(range(2, 5, 1))
		batch_size_list = list(range(40, 130, 10))
		weight_decay_list = [0.0001, 0.001, 0.01, 0.1, 1, 10]
		prediction_keep_rate_list = list(np.arange(0.7, 0.95, 0.05))
		keep_rate_list = list(np.arange(0.7, 0.95, 0.05))
		learning_rate_list = list(np.random.uniform(0.0001, 0.002, 100))
		RNN_init_state_noise_stddev_list = list(np.random.uniform(0.001, 0.9, 100))
		self.hyper_config = HyperParameterConfig()
		self.hyper_config.CNN_RNN_2()
		result_summary = OrderedDict()

		def delete_dir(dir_index, obj):
			key_paths = self._find_in_obj(obj, 'testing_accurcy')
			key_paths = list(key_paths)
			# print(key_paths)
			accuracy_threshold = 0.7

			for key in key_paths:
				value = reduce(operator.getitem, key, obj)
				# key_value = [(ele, reduce(operator.getitem, ele, obj)) for ele in key]
				# print(value)
				if accuracy_threshold > value:
					delete_dir = os.path.join(
						self.basic_result_path,
						'_' + str(dir_index))
					shutil.rmtree(delete_dir)
					print('delete dir!')
					break

		def random_choise(hyper_list):
			return random.choice(hyper_list)

		def set_attr(set_attribution, para):
			setattr(self.hyper_config, set_attribution, para)

		def run_random_search(run_index):
			'''
			fully_connected = random_choise(fully_connected_list)
			set_attr('fully_connected_units', fully_connected)

			prediction_nodes = random_choise(prediction_nodes_list)
			set_attr('prediction_layer_1_uints', prediction_nodes)

			RNN_nodes = random_choise(RNN_nodes_list)
			set_attr('RNN_hidden_node_size', RNN_nodes)

			RNN_layer = random_choise(RNN_layer_list)
			set_attr('RNN_num_layers', RNN_layer)

			batch_size = random_choise(batch_size_list)
			set_attr('batch_size', batch_size)

			weight_decay = random_choise(weight_decay_list)
			set_attr('weight_decay', weight_decay)

			prediction_keep_rate = random_choise(prediction_keep_rate_list)
			set_attr('prediction_layer_keep_rate', prediction_keep_rate)

			keep_rate = random_choise(keep_rate_list)
			set_attr('keep_rate', keep_rate)

			learning_rate = random_choise(learning_rate_list)
			set_attr('learning_rate', learning_rate)

			RNN_init_state_noise_stddev = random_choise(RNN_init_state_noise_stddev_list)
			set_attr('RNN_init_state_noise_stddev', RNN_init_state_noise_stddev)
			'''
			self._search_CNN_RNN_2()

			save_model_path = os.path.join(
				self.basic_result_path,
				'_' + str(run_index),
				self.search_task_name + '.ckpt')

			result_path = os.path.join(
				self.basic_result_path,
				'_' + str(run_index))

			model_path = {
				'reload_path': './output_model/CNN_RNN_test.ckpt',
				'save_path': save_model_path,
				'result_path': result_path
			}
			result = self._run_neural_network(model_path, self.hyper_config, CNN_RNN_2)
			return result

		self.search_task_name = task_name
		self.basic_result_path = './result/'
		self.basic_result_path = os.path.join(self.basic_result_path, self.search_task_name)
		report_path = os.path.join(self.basic_result_path, self.search_task_name + '.txt')
		if not os.path.exists(self.basic_result_path):
			os.makedirs(self.basic_result_path)

		for run_index in range(200):
			result_summary[str(run_index)] = run_random_search(run_index)
			# delete_dir(run_index, result_summary[str(run_index)])
			if run_index % 5 == 0 and run_index is not 0:
				max_pairs = self._find_highest_accu(result_summary)
				with open(report_path, 'w') as outfile:
					json_string = json.dumps(result_summary, sort_keys=False, indent=4)
					outfile.write(json_string + '\n')
					for max_pair in max_pairs:
						line = 'value:{} in task: {} {}:{}'.format(
							max_pair[0][0],
							max_pair[0][1],
							max_pair[0][2],
							max_pair[1])
						print(line)
						outfile.write(line + '\n')

		print('search finish!!')
		max_pairs = self._find_highest_accu(result_summary)
		with open(report_path, 'w') as outfile:
			json_string = json.dumps(result_summary, sort_keys=False, indent=4)
			# print(json_string)
			outfile.write(json_string + '\n')

			for max_pair in max_pairs:
				line = 'value:{} in task: {} {}:{}'.format(
					max_pair[0][0],
					max_pair[0][1],
					max_pair[0][2],
					max_pair[1])
				print(line)
				outfile.write(line + '\n')

	def _run_grid_search(self, iterables_paras, set_attr):
		result_summary = OrderedDict()

		for para in iterables_paras:
			setattr(self.hyper_config, set_attr, para)
			save_model_path = os.path.join(
				self.basic_result_path + '_' + str(para),
				self.search_task_name + '.ckpt')

			result_path = os.path.join(
				self.basic_result_path,
				'_' + str(para))

			report_path = os.path.join(self.basic_result_path, self.search_task_name + '.txt')
			# print(save_model_path)
			# print(result_path)
			model_path = {
				'reload_path': './output_model/CNN_RNN_test.ckpt',
				'save_path': save_model_path,
				'result_path': result_path
			}
			result = self._run_CNN_RNN(model_path, self.hyper_config)
			result_summary[str(para)] = result
		print('search finish!!')

		max_pairs = self._find_highest_accu(result_summary)

		with open(report_path, 'w') as outfile:
			json_string = json.dumps(result_summary, sort_keys=False, indent=4)
			# print(json_string)
			outfile.write(json_string + '\n')

			for max_pair in max_pairs:
				line = 'value:{} in task: {} {}:{}'.format(
					max_pair[0][0],
					max_pair[0][1],
					max_pair[0][2],
					max_pair[1])
				print(line)
				outfile.write(line + '\n')

	def _run_neural_network(self, model_path, config, neural_network):
		input_data_shape = [self.X_array.shape[1], self.X_array.shape[2], self.X_array.shape[3], self.X_array.shape[4]]
		output_data_shape = [self.Y_array.shape[1], self.Y_array.shape[2], self.Y_array.shape[3], 1]
		cnn_rnn = neural_network(input_data_shape, output_data_shape, config)
		cnn_rnn.create_MTL_task(self.X_array, self.Y_array[:, :, :, :, 0, np.newaxis], 'min_traffic')
		cnn_rnn.create_MTL_task(self.X_array, self.Y_array[:, :, :, :, 1, np.newaxis], 'avg_traffic')
		cnn_rnn.create_MTL_task(self.X_array, self.Y_array[:, :, :, :, 2, np.newaxis], 'max_traffic')
		return cnn_rnn.start_MTL_train(model_path, reload=False)


if __name__ == '__main__':

	config = HyperParameterConfig()
	# con_key_val = config.get_variable()
	config.save_json()
