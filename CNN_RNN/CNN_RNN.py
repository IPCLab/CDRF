import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import os


class Tf_Utility:
	def weight_variable(self, shape, name):
		# initial = tf.truncated_normal(shape, stddev=0.1)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

	def bias_variable(self, shape, name):
		# initial = tf.random_normal(shape)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

	def write_to_Tfrecord(self, X_array, Y_array, filename):
		writer = tf.python_io.TFRecordWriter(filename)
		for index, each_record in enumerate(X_array):
			tensor_record = each_record.astype(np.float32).tobytes()
			tensor_result = Y_array[index].astype(np.float32).tobytes()
			# print('in _write_to_Tfrecord',X_array.shape,Y_array.shape)
			example = tf.train.Example(features=tf.train.Features(feature={
				'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
				'record': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_record])),
				'result': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_result]))
			}))

			writer.write(example.SerializeToString())
		writer.close()

	def read_data_from_Tfrecord(
		self,
		filename,
		input_temporal,
		input_vertical,
		input_horizontal,
		input_channel,
		Y_temporal,
		Y_vertical,
		Y_horizontal,
		Y_channel):
		filename_queue = tf.train.string_input_producer([filename])
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
			serialized_example,
			features={
				'index': tf.FixedLenFeature([], tf.int64),
				'record': tf.FixedLenFeature([], tf.string),
				'result': tf.FixedLenFeature([], tf.string)
			})
		index = features['index']
		record = tf.decode_raw(features['record'], tf.float32)
		result = tf.decode_raw(features['result'], tf.float32)
		record = tf.reshape(record, [
			input_temporal,
			input_vertical,
			input_horizontal,
			input_channel])
		result = tf.reshape(result, [
			Y_temporal,
			Y_vertical,
			Y_horizontal,
			Y_channel])

		return index, record, result

	def read_all_data_from_Tfreoced(
		self,
		filename,
		input_temporal,
		input_vertical,
		input_horizontal,
		input_channel,
		Y_temporal,
		Y_vertical,
		Y_horizontal,
		Y_channel):
		record_iterator = tf.python_io.tf_record_iterator(path=filename)
		record_list = []
		result_list = []
		for string_record in record_iterator:
			example = tf.train.Example()
			example.ParseFromString(string_record)
			index = example.features.feature['index'].int64_list.value[0]
			record = example.features.feature['record'].bytes_list.value[0]
			result = example.features.feature['result'].bytes_list.value[0]
			record = np.fromstring(record, dtype=np.float32)
			record = record.reshape((
				input_temporal,
				input_vertical,
				input_horizontal,
				input_channel))

			result = np.fromstring(result, dtype=np.float32)
			result = result.reshape((
				Y_temporal,
				Y_vertical,
				Y_horizontal,
				Y_channel))
			record_list.append(record)
			result_list.append(result)

		record = np.stack(record_list)
		result = np.stack(result_list)
		return index, record, result

	def save_model(self, sess, saver, model_path):
		# model_path = './output_model/CNN_RNN.ckpt'
		print('saving model.....')
		try:
			save_path = saver.save(sess, model_path)
			# self.pre_train_saver.save(sess, model_path + '_part')
		except Exception:
			save_path = saver.save(sess, './output_model/temp.ckpt')
		finally:
			print('save_path:{}'.format(save_path))

	def reload_model(self, sess, saver, model_path):
		print('reloading model {}.....'.format(model_path))
		saver.restore(sess, model_path)

	def print_all_tensor(self):
		graph = tf.get_default_graph()
		all_vars = [n.name for n in graph.as_graph_def().node]
		for var_s in all_vars:
			print(var_s)

	def print_all_trainable_var(self):
		vars_list = tf.trainable_variables()
		for var_s in vars_list:
			print(var_s)


class Multitask_Neural_Network(Tf_Utility):

	def build_MTL(self, input_data_shape, output_data_shape):
		tf.reset_default_graph()
		tl.layers.clear_layers_name()
		self.shuffle_min_after_dequeue = 600
		self.shuffle_capacity = self.shuffle_min_after_dequeue + 3 * self.batch_size

		self.input_temporal = input_data_shape[0]
		self.input_vertical = input_data_shape[1]
		self.input_horizontal = input_data_shape[2]
		self.input_channel = input_data_shape[3]

		self.output_temporal = output_data_shape[0]
		self.output_vertical = output_data_shape[1]
		self.output_horizontal = output_data_shape[2]
		self.output_channel = 1
		self.predictor_output = self.output_temporal * self.output_vertical * self.output_horizontal * self.output_channel

		self.Xs = tf.placeholder(tf.float32, shape=[
			None, self.input_temporal, self.input_vertical, self.input_horizontal, self.input_channel], name='Input_x')
		self.Ys = tf.placeholder(tf.float32, shape=[
			None, self.output_temporal, self.output_vertical, self.output_horizontal, 1], name='Input_y')
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		self.add_noise = tf.placeholder(tf.bool, name='add_noise')
		self.RNN_init_state = tf.placeholder(tf.float32, [self.RNN_num_layers, 2, None, self.RNN_hidden_node_size])  # 2: hidden state and cell state
		self.multi_task_dic = {}

	def build_flatten_layer(self, tl_input):

		flat_tl = tl.layers.FlattenLayer(tl_input, name='flatten_layer')
		network = tl.layers.DenseLayer(flat_tl, W_init=self.fully_connected_W_init, n_units=self.fully_connected_units, act=lambda x: tl.act.lrelu(x, 0.2), name='fully_connect_1')
		network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
		self.tl_share_output = network

	def create_MTL_task(self, input_x, input_y, task_name, loss_type='MSE'):
		self.multi_task_dic[task_name] = self.__create_MTL_output(self.tl_share_output, self.Ys, task_name, loss_type='MSE')
		self.__set_training_data(input_x, input_y, task_name)
		self.saver = tf.train.Saver()

	def __create_MTL_output(self, tl_input, y, scope_name, loss_type='MSE'):
		def get_l2_list():
			# print('get_l2_list:')
			var_list = []
			exclude_list = ['LSTMCell/B', 'regression_op/b', 'b_conv2d']
			for v in tf.trainable_variables():
				if any(x in v.name for x in exclude_list):
					continue
				if 'prediction_layer' in v.name and scope_name not in v.name:
					continue
				# print(v)
				var_list.append(v)
			return var_list

		def get_trainable_var():
			# print('get_trainable_var:')
			var_list = []
			for v in tf.trainable_variables():
				if 'prediction_layer' in v.name:
					if scope_name not in v.name:
						continue
				# print(v)
				var_list.append(v)
			return var_list

		def get_prediction_layer_var():
			var_list = []
			for v in tf.trainable_variables():
				if 'prediction_layer' in v.name:
					if scope_name in v.name:
						var_list.append(v)
						# print(v)
			return var_list

		with tf.variable_scope('prediction_layer'):
			with tf.variable_scope(scope_name):

				tl_input = tl.layers.BatchNormLayer(tl_input, is_train=False, name='batch_norm')
				tl_regression = tl.layers.DenseLayer(tl_input, W_init=self.prediction_layer_1_W_init, n_units=self.prediction_layer_1_uints, act=lambda x: tl.act.lrelu(x, 0.2), name='regression_op_1')
				tl_regression = tl.layers.DropoutLayer(tl_regression, keep=self.prediction_layer_keep_rate, name='drop_1')
				tl_regression = tl.layers.DenseLayer(tl_input, W_init=self.prediction_layer_2_W_init, n_units=self.predictor_output, act=tl.activation.identity, name='regression_op_2')
				tl_regression = tl.layers.DropoutLayer(tl_regression, keep=self.prediction_layer_keep_rate, name='drop_2')
				tl_output = tl_regression
				regression_output = tl_output.outputs
				# print('regression_output shape {}'.format(regression_output.get_shape().as_list()))
				output = tf.reshape(regression_output, [-1, self.output_temporal, self.output_vertical, self.output_horizontal, self.output_channel], name='output_layer')

				cross_entropy = tf.nn.softmax_cross_entropy_with_logits(output, y, name='corss_entropy_op')
				MSE = tf.reduce_mean(tf.pow(output - y, 2), name='MSE_op')
				RMSE = tf.sqrt(tf.reduce_mean(tf.pow(output - y, 2)))
				MAE = tf.reduce_mean(tf.abs(output - y))
				# MAPE = tf.reduce_mean(tf.abs(tf.divide(y - output, y)), name='MAPE_OP')
				MAPE = tf.reduce_mean(tf.divide(tf.abs(y - output), tf.reduce_mean(y)), name='MAPE_OP')
				L2_list = get_l2_list()
				L2_loss = self.__L2_norm(L2_list)

				if loss_type == 'cross_entropy':
					prediction_softmax = tf.nn.softmax(output)
					output = tf.argmax(prediction_softmax, 1)
					correct_prediciton = tf.equal(tf.argmax(prediction_softmax, 1), tf.argmax(y, 1))
					accuracy = tf.reduce_mean(tf.cast(correct_prediciton, tf.float32))
					cost = tf.add(cross_entropy, L2_loss * self.weight_decay, name='cost_op')
				else:
					cost = tf.add(MSE, L2_loss * self.weight_decay, name='cost_op')
					accuracy = 1 - MAPE
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				opt_vars = get_trainable_var()
				gvs = optimizer.compute_gradients(cost, var_list=opt_vars)
				capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if grad is not None]
				optimizer_op = optimizer.apply_gradients(capped_gvs)

				optimizer_predict = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				only_prediction_opt_vars = get_prediction_layer_var()
				gvs_predict = optimizer_predict.compute_gradients(MAE, var_list=only_prediction_opt_vars)
				capped_gvs_predict = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs_predict if grad is not None]
				optimizer_op_predict = optimizer.apply_gradients(capped_gvs_predict)
				task_dic = {
					'output': output,
					'optomizer': optimizer_op,
					'prediction_optimizer': optimizer_op_predict,
					'tl_output': tl_output,
					'cost': cost,
					'L2_loss': L2_loss,
					'MSE': MSE,
					'MAE': MAE,
					'MAPE': MAPE,
					'RMSE': RMSE,
					'cross_entropy': cross_entropy,
					'accurcy': accuracy,
					'training_accurcy_history': [],
					'testing_accurcy_history': [],
					'training_MSE_history': [],
					'testing_MSE_history': [],
					'training_temp_loss': 0
				}
				return task_dic

	def parse_pooling(self, type_fn='max_pool'):
			if type_fn == 'max_pool':
				func = tf.nn.max_pool
			else:
				func = tf.nn.avg_pool
			return func

	def parse_activation(self, type_fn='relu'):
		if type_fn == 'relu':
			func = tf.nn.relu
		else:
			func = tf.nn.relu
		return func

	def parse_initializer_method(self, type_fn='xavier'):
		if type_fn:
			func = tf.contrib.layers.xavier_initializer_conv2d()
		else:
			func = tf.truncated_normal_initializer(stddev=0.1)

		return func

	def parse_RNN_cell(self, type_fn='LSTMcell'):
		if type_fn:
			cell = tf.nn.rnn_cell.LSTMCell
		else:
			cell = tf.nn.rnn_cell.BasicLSTMCell

		return cell

	def __summarized_report(self):
		task_keys = self.multi_task_dic.keys()
		task_keys = sorted(task_keys)
		summary_dic = {}
		for key in task_keys:
			task_summ_dict = {}
			train_MSE = self.multi_task_dic[key]['training_MSE_history'][-1]
			train_accu = self.multi_task_dic[key]['training_accurcy_history'][-1]
			test_MSE = self.multi_task_dic[key]['testing_MSE_history'][-1]
			test_accu = self.multi_task_dic[key]['testing_accurcy_history'][-1]

			task_summ_dict['training_MSE'] = train_MSE
			task_summ_dict['training_accurcy'] = train_accu
			task_summ_dict['testing_MSE'] = test_MSE
			task_summ_dict['testing_accurcy'] = test_accu
			summary_dic[key] = task_summ_dict
		return summary_dic

	def print_all_layers(self):
		self.tl_share_output.print_layers()
		# print(self.tl_share_output.all_layers)

	def print_all_variables(self):
		self.tl_share_output.print_params()

	def __L2_norm(self, var_list):
		L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list])
		return L2_loss

	def __set_training_data(self, input_x, input_y, task_name):

		# print('input_x shape:{}'.format(input_x.shape))
		# print('input_y shape:{}'.format(input_y.shape))
		self.Y_temporal = input_y.shape[1]
		self.Y_vertical = input_y.shape[2]
		self.Y_horizontal = input_y.shape[3]
		self.Y_channel = input_y.shape[4]
		# input_x, self.mean, self.std = self.feature_normalize_input_data(input_x)
		self.mean = 0
		self.std = 1
		X_data = input_x
		Y_data = input_y

		# Y_data = Y_data[:,np.newaxis]
		# print(X_data[1,0,0,0,-1],Y_data[0,0,0,0,-1])

		training_X = X_data[0:int(9 * X_data.shape[0] / 10)]
		training_Y = Y_data[0:int(9 * Y_data.shape[0] / 10)]
		# training_X = X_data  # todo
		# training_Y = Y_data  # todo
		testing_X = X_data[int(9 * X_data.shape[0] / 10):]
		testing_Y = Y_data[int(9 * Y_data.shape[0] / 10):]

		training_file = task_name + '_training.tfrecoeds'
		testing_file = task_name + '_testing.tfrecoeds'

		print('training X shape:{}, training Y shape:{}'.format(
			training_X.shape, training_Y.shape))
		self.write_to_Tfrecord(training_X, training_Y, training_file)
		self.write_to_Tfrecord(testing_X, testing_Y, testing_file)
		self.training_data_number = training_X.shape[0]
		self.multi_task_dic[task_name]['training_file'] = training_file
		self.multi_task_dic[task_name]['testing_file'] = testing_file

		training_data = self.read_data_from_Tfrecord(
			self.multi_task_dic[task_name]['training_file'],
			self.input_temporal,
			self.input_vertical,
			self.input_horizontal,
			self.input_channel,
			self.Y_temporal,
			self.Y_vertical,
			self.Y_horizontal,
			self.Y_channel)
		batch_tuple_OP = tf.train.shuffle_batch(
			training_data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)
		batch_without_shuffle_OP = tf.train.batch(
			training_data,
			batch_size=self.batch_size)

		self.multi_task_dic[task_name]['shuffle_batch_OP'] = batch_tuple_OP
		self.multi_task_dic[task_name]['batch_OP'] = batch_without_shuffle_OP

	def _MTL_testing_data(self, sess, test_x, test_y, task_name):
		task_dic = self.multi_task_dic[task_name]
		predict_list = []
		cum_MSE = 0
		cum_accu = 0
		batch_num = test_x.shape[0] // self.batch_size
		for batch_index in range(batch_num):
			dp_dict = tl.utils.dict_to_one(task_dic['tl_output'].all_drop)
			batch_x = test_x[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
			batch_y = test_y[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
			feed_dict = {
				self.Xs: batch_x,
				self.Ys: batch_y,
				self.keep_prob: 1,
				self.add_noise: 0}
			feed_dict.update(dp_dict)
			with tf.device('/gpu:0'):
				MSE, accu, predict = sess.run([task_dic['MSE'], task_dic['accurcy'], task_dic['output']], feed_dict=feed_dict)
			'''
			for i in range(10, 15):
				for j in range(predict.shape[1]):
					print('batch index: {} predict:{:.4f} real:{:.4f}'.format(batch_index, predict[i, j, 0, 0, 0], batch_y[i, j, 0, 0, 0]))
			print()
			'''
			for predict_element in predict:
				predict_list.append(predict_element)
			cum_MSE += MSE
			cum_accu += accu
		return cum_MSE / batch_num, cum_accu / batch_num, np.stack(predict_list)

	def save_result_report(self, dir_name='./result/temp'):
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		with open(os.path.join(dir_name, 'report.txt'), 'w') as f:
			task_keys = self.multi_task_dic.keys()
			task_keys = sorted(task_keys)

			record_len = len(self.multi_task_dic[task_keys[0]]['training_accurcy_history'])
			for num in range(record_len):
				for key in task_keys:
					line = key + ': \n'
					line += ' train_MSE: ' + str(self.multi_task_dic[key]['training_MSE_history'][num])
					line += ' train_accu: ' + str(self.multi_task_dic[key]['training_accurcy_history'][num])
					line += ' test_MES: ' + str(self.multi_task_dic[key]['testing_MSE_history'][num])
					line += ' test_accu: ' + str(self.multi_task_dic[key]['testing_accurcy_history'][num])
					f.write(line + '\n')
				f.write('\n')

	def __save_hyperparameter(self, config, dir_name='./result/temp'):
			if not os.path.exists(dir_name):
				os.makedirs(dir_name)

			config.save_json(os.path.join(dir_name, 'config.json'))
			'''
			sort_key_val = [(k, config[k]) for k in sorted(config.keys())]

			with open(dir_name + 'config.txt', 'w') as f:
				for element in sort_key_val:
					line = str(element[0]) + ': ' + str(element[1])
					f.write(line + '\n')
			'''
	def _plot_predict_vs_real(self, fig_instance, task_name, testing_y, testing_predict_y, training_y, training_predict_y):

		ax_1 = fig_instance.add_subplot(3, 1, 1)
		ax_2 = fig_instance.add_subplot(3, 1, 2)
		ax_3 = fig_instance.add_subplot(3, 1, 3)
		ax_1.cla()
		ax_2.cla()
		ax_3.cla()

		ax_1.plot(testing_y, label='real', marker='.')
		ax_1.plot(testing_predict_y, label='predict', marker='.')
		ax_1.set_title(task_name + ' testing data')
		ax_1.grid()
		ax_1.legend()

		ax_2.plot(training_y, label='real', marker='.')
		ax_2.plot(training_predict_y, label='predict', marker='.')
		ax_2.set_title(task_name + ' training data')
		ax_2.grid()
		ax_2.legend()

		ax_3.plot(self.multi_task_dic[task_name]['training_MSE_history'], 'g-', label=task_name + ' training losses')
		ax_3.plot(self.multi_task_dic[task_name]['testing_MSE_history'], 'b-', label=task_name + ' testing losses')
		ax_3.set_title(task_name + 'loss')
		ax_3.grid()
		ax_3.legend()
		# ax.draw()
		plt.pause(0.001)

	def start_STL_predict(self, testing_x, testing_y, model_path, task_name):

		print('testing_x shape {}'.format(testing_x.shape))
		print('testing_y shape {}'.format(testing_y.shape))
		# self.print_all_layers()
		with tf.Session() as sess:
			self.reload_model(sess, self.saver, model_path['reload_path'])
			testing_loss, testing_accu, prediction = self._MTL_testing_data(sess, testing_x, testing_y, task_name)

			print('preddict finished!')
			print('task {}: accu:{} MSE:{}'.format(task_name, testing_accu, testing_loss))

		return prediction

	def start_MTL_predict(self, testing_x, testing_y, model_path):
			def get_multi_task_batch(batch_x, batch_y):
				batch_y = np.transpose(batch_y, [4, 0, 1, 2, 3])
				batch_y = np.expand_dims(batch_y, axis=5)
				return batch_x, batch_y

			print('input_x shape {}'.format(testing_x.shape))
			print('input_y shape {}'.format(testing_y.shape))
			testing_x, testing_y = get_multi_task_batch(testing_x, testing_y)
			# tf.reset_default_graph()
			# tf.train.import_meta_graph(model_path['reload_path'] + '.meta')

			self.print_all_layers()
			with tf.Session() as sess:
				self.reload_model(sess, self.saver, model_path['reload_path'])
				testing_loss_min, testing_accu_min, prediction_min = self._MTL_testing_data(sess, testing_x, testing_y[0], 'min_traffic')
				testing_loss_avg, testing_accu_avg, prediction_avg = self._MTL_testing_data(sess, testing_x, testing_y[1], 'avg_traffic')
				testing_loss_max, testing_accu_max, prediction_max = self._MTL_testing_data(sess, testing_x, testing_y[2], 'max_traffic')
			print('preddict finished!')
			print('task Min: accu:{} MSE:{}'.format(testing_accu_min, testing_loss_min))
			print('task avg: accu:{} MSE:{}'.format(testing_accu_avg, testing_loss_avg))
			print('task Max: accu:{} MSE:{}'.format(testing_accu_max, testing_loss_max))

			return [prediction_min, prediction_avg, prediction_max]

	def run_multi_task(self, sess, task_name='', optimizer='optomizer'):
		def run_task_optimizer(sess, batch_x, batch_y, task_name, optimizer='optomizer'):
			task_dic = self.multi_task_dic[task_name]
			feed_dict = {
				self.Xs: batch_x,
				self.Ys: batch_y,
				self.keep_prob: 0.85,
				self.add_noise: 1}
			feed_dict.update(task_dic['tl_output'].all_drop)
			_, cost, L2_loss = sess.run([task_dic[optimizer], task_dic['cost'], task_dic['L2_loss']], feed_dict=feed_dict)

			return cost, L2_loss
		task = self.multi_task_dic[task_name]
		training_batch_op = task['shuffle_batch_OP']
		index, batch_x, batch_y = sess.run(training_batch_op)
		# batch_x, batch_y = get_multi_task_batch(batch_x, batch_y)
		cost, L2 = run_task_optimizer(sess, batch_x, batch_y, task_name)
		task['training_temp_loss'] += cost

	def run_task_evaluate(self, sess, fig, epoch, display_step=50, task_name=''):
			task = self.multi_task_dic[task_name]
			index, testing_X, testing_Y = self.read_all_data_from_Tfreoced(
				task['testing_file'],
				self.input_temporal,
				self.input_vertical,
				self.input_horizontal,
				self.input_channel,
				self.Y_temporal,
				self.Y_vertical,
				self.Y_horizontal,
				self.Y_channel)
			index, batch_x_sample, batch_y_sample = sess.run(task['batch_OP'])
			# batch_x_sample, batch_y_sample = get_multi_task_batch(batch_x_sample, batch_y_sample)
			# testing_X, testing_Y = get_multi_task_batch(testing_X, testing_Y)
			testing_loss, testing_accu, prediction = self._MTL_testing_data(sess, testing_X, testing_Y, task_name)
			training_loss_nodrop, training_accu, train_prediction = self._MTL_testing_data(sess, batch_x_sample, batch_y_sample, task_name)

			task['training_temp_loss'] /= display_step

			self.multi_task_dic[task_name]['testing_MSE_history'].append(testing_loss)
			self.multi_task_dic[task_name]['training_MSE_history'].append(training_loss_nodrop)
			self.multi_task_dic[task_name]['testing_accurcy_history'].append(testing_accu)
			self.multi_task_dic[task_name]['training_accurcy_history'].append(training_accu)
			print('task:{} epoch:{} training_cost:{:.4f} trainin_MSE(nodrop):{:.4f} training_accu:{:.4f} testing_MSE(nodrop):{:.4f} testing_accu:{:.4f}'.format(
				task_name,
				epoch,
				task['training_temp_loss'],
				training_loss_nodrop,
				training_accu,
				testing_loss,
				testing_accu))
			self._plot_predict_vs_real(
				fig,
				task_name,
				testing_Y[:100, 0, 0, 0, 0],
				prediction[:100, 0, 0, 0, 0],
				batch_y_sample[:100, 0, 0, 0, 0],
				train_prediction[:100, 0, 0, 0, 0])
			task['training_temp_loss'] = 0

	def start_MTL_train(self, model_path, reload=False):
		display_step = 50
		epoch_his = []
		plt.ion()
		# loss_fig = plt.figure(0)
		min_fig = plt.figure(1)
		avg_fig = plt.figure(2)
		max_fig = plt.figure(3)
		_10_mins_fig = plt.figure(4)

		# model_base_name = os.path.basename(model_path['save_path'])
		# model_base_name = os.path.splitext(model_base_name)[0]
		# dir_name = './result/' + model_base_name + '/'
		result_path = model_path['result_path']

		def save_figure(dir_name='./result/temp'):
			min_fig.set_size_inches(12, 9)
			avg_fig.set_size_inches(12, 9)
			max_fig.set_size_inches(12, 9)
			min_fig.savefig(os.path.join(dir_name, 'min.png'), dpi=100)
			avg_fig.savefig(os.path.join(dir_name, 'avg.png'), dpi=100)
			max_fig.savefig(os.path.join(dir_name, 'max.png'), dpi=100)

		def get_multi_task_batch(batch_x, batch_y):
			batch_y = np.transpose(batch_y, [4, 0, 1, 2, 3])
			batch_y = np.expand_dims(batch_y, axis=5)
			return batch_x, batch_y

		def early_stop(epoch, stop_type=1):
			task_keys = self.multi_task_dic.keys()
			task_keys = sorted(task_keys)
			Flag = False
			'''
			if epoch >= 600:
				if self.multi_task_dic['max_traffic']['testing_accurcy_history'][-1] > 0.75:
					if self.multi_task_dic['avg_traffic']['testing_accurcy_history'][-1] > 0.75:
						Flag = True
				if self.multi_task_dic['avg_traffic']['testing_accurcy_history'][-1] > 0.8:
					if self.multi_task_dic['max_traffic']['testing_accurcy_history'][-1] > 0.73:
						Flag = True

			if epoch > 1000:
				if self.multi_task_dic['max_traffic']['testing_accurcy_history'][-1] < 0.68:
					Flag = True
				if self.multi_task_dic['avg_traffic']['testing_accurcy_history'][-1] < 0.7:
					Flag = True
				if self.multi_task_dic['min_traffic']['testing_accurcy_history'][-1] < 0.7:
					Flag = True
			'''
			if epoch >= 500:
				if self.multi_task_dic['max_traffic']['testing_accurcy_history'][-1] > 0.74:
					if self.multi_task_dic['avg_traffic']['testing_accurcy_history'][-1] > 0.74:
						Flag = True
				if self.multi_task_dic['max_traffic']['testing_accurcy_history'][-1] > 0.76:
					if self.multi_task_dic['avg_traffic']['testing_accurcy_history'][-1] > 0.70:
						Flag = True
				if self.multi_task_dic['avg_traffic']['testing_accurcy_history'][-1] > 0.8:
					if self.multi_task_dic['max_traffic']['testing_accurcy_history'][-1] > 0.70:
						Flag = True

			if epoch >= 800:
				if self.multi_task_dic['max_traffic']['testing_accurcy_history'][-1] > 0.70:
					if self.multi_task_dic['avg_traffic']['testing_accurcy_history'][-1] > 0.70:
						Flag = True
			'''
			for key in task_keys:
				test_accu = self.multi_task_dic[key]['testing_accurcy_history'][-1]
				if stop_type:
					if test_accu < 0.6:
						Flag = True
				else:
					if test_accu < 0.74:
						Flag = False
						break
					else:
						Flag = True
			'''
			return Flag

		def _plot_loss_rate(epoch_his):
			ax_1 = loss_fig.add_subplot(3, 1, 1)
			ax_2 = loss_fig.add_subplot(3, 1, 2)
			ax_3 = loss_fig.add_subplot(3, 1, 3)
			ax_1.cla()
			ax_2.cla()
			ax_3.cla()
			ax_1.plot(self.multi_task_dic['min_traffic']['training_cost_history'], 'g-', label='min training losses')
			ax_1.plot(self.multi_task_dic['min_traffic']['testing_cost_history'], 'b-', label='min testing losses')
			ax_1.legend()
			ax_2.plot(self.multi_task_dic['avg_traffic']['training_cost_history'], 'g-', label='avg training losses')
			ax_2.plot(self.multi_task_dic['avg_traffic']['testing_cost_history'], 'b-', label='avg testing losses')
			ax_2.legend()
			ax_3.plot(self.multi_task_dic['max_traffic']['training_cost_history'], 'g-', label='max training losses')
			ax_3.plot(self.multi_task_dic['max_traffic']['testing_cost_history'], 'b-', label='max testing losses')
			ax_3.legend()
			plt.pause(0.001)

		self.__save_hyperparameter(self.hyper_config, result_path)

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess, coord=coord)
			tf.summary.FileWriter('logs/', sess.graph)
			if reload:
				self.reload_model(sess, self.saver, model_path['reload_path'])
			else:
				sess.run(tf.global_variables_initializer())
			with tf.device('/gpu:0'):
				for epoch in range(self.iter_epoch):
					# run_multi_task(sess, 'min_traffic', 'prediction_optimizer')
					# run_multi_task(sess, 'avg_traffic', 'prediction_optimizer')
					# run_multi_task(sess, 'max_traffic', 'prediction_optimizer')

					self.run_multi_task(sess, 'min_traffic')
					self.run_multi_task(sess, 'avg_traffic')
					self.run_multi_task(sess, 'max_traffic')
					# run_multi_task(sess, '10_mins')

					if epoch % display_step == 0 and epoch is not 0:
						self.run_task_evaluate(sess, min_fig, epoch, task_name='min_traffic')
						self.run_task_evaluate(sess, avg_fig, epoch, task_name='avg_traffic')
						self.run_task_evaluate(sess, max_fig, epoch, task_name='max_traffic')
						# run_task_evaluate(sess, _10_mins_fig, epoch, task_name='10_mins')
						print()
						epoch_his.append(epoch)
						# _plot_loss_rate(epoch_his)

					if epoch % 500 == 0 and epoch is not 0:
						self.save_model(sess, self.saver, model_path['save_path'])
						self.save_result_report(result_path)
						save_figure(result_path)

					# if epoch >= 400:
						# if epoch % 100 == 0 and epoch is not 0:
					flag = early_stop(epoch, 0)
					if flag:
						break
			coord.request_stop()
			coord.join(treads)
			print('training finished!')
			self.save_model(sess, self.saver, model_path['save_path'])
			self.save_result_report(result_path)
			save_figure(result_path)
		plt.ioff()
		plt.show()
		return self.__summarized_report()

	def start_STL_train(self, model_path, task_name, reload=False):
		def early_stop(epoch, task_name):
			task = self.multi_task_dic[task_name]
			Flag = False
			if epoch >= 100:
				if task['testing_accurcy_history'][-1] > 0.7:
						Flag = True
			return Flag
		display_step = 50
		epoch_his = []
		plt.ion()
		fig = plt.figure(task_name)
		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess, coord=coord)
			tf.summary.FileWriter('logs/', sess.graph)

			if reload:
					self.reload_model(sess, self.saver, model_path['reload_path'])
			else:
				sess.run(tf.global_variables_initializer())
			with tf.device('/gpu:0'):
				for epoch in range(self.iter_epoch):
					self.run_multi_task(sess, task_name)
					if epoch % display_step == 0 and epoch is not 0:
						self.run_task_evaluate(sess, fig, epoch, task_name=task_name)
						# print()
						epoch_his.append(epoch)

					if epoch % 500 == 0 and epoch is not 0:
						self.save_model(sess, self.saver, model_path['save_path'])

					# flag = early_stop(epoch, task_name)
					# if flag:
						# break
			coord.request_stop()
			coord.join(treads)
			print('training finished!')
			self.save_model(sess, self.saver, model_path['save_path'])
			# self.save_result_report(result_path)
			plt.ioff()
		# plt.show()—


class CNN_3D(Multitask_Neural_Network):
	def __init__(self, input_data_shape, output_data_shape, config):
		# super().__init__(input_data_shape, output_data_shape, config)

		tl_output = self.__build_3D_CNN(self.Xs)
		self.build_flatten_layer(tl_output)

	def __build_3D_CNN(self, Xs):
		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN'):
				CNN_input = tf.reshape(input_X, [-1, self.input_temporal, self.input_vertical, self.input_horizontal, self.input_channel])
				# print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_1')
				network = tl.layers.Conv3dLayer(
					network,
					act=tf.nn.relu,
					shape=[3, 5, 5, 1, 32],
					strides=[1, 2, 2, 2, 1],
					padding='SAME',
					name='Con3d_layer_1')
				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 2, 2, 2, 1],
					strides=[1, 1, 2, 2, 1],
					padding='SAME',
					pool=tf.nn.max_pool3d,
					name='pooling_layer_1')

				network = tl.layers.Conv3dLayer(
					network,
					act=tf.nn.relu,
					shape=[3, 5, 5, 32, 64],
					strides=[1, 2, 2, 2, 1],
					padding='SAME',
					name='Con3d_layer_2')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 2, 2, 2, 1],
					strides=[1, 1, 1, 1, 1],
					padding='SAME',
					pool=tf.nn.avg_pool3d,
					name='pooling_layer_2')
				network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_2')

				network = tl.layers.Conv3dLayer(
					network,
					act=tf.nn.relu,
					shape=[3, 5, 5, 64, 64],
					strides=[1, 2, 2, 2, 1],
					padding='SAME',
					name='Con3d_layer_3')
				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 2, 2, 2, 1],
					strides=[1, 1, 1, 1, 1],
					padding='SAME',
					pool=tf.nn.avg_pool3d,
					name='pooling_layer_3')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
			return network

		tl_3D_CNN_output = build_CNN_network(Xs)
		return tl_3D_CNN_output


class RNN(Multitask_Neural_Network):
	def __init__(self, input_data_shape, output_data_shape, config):
		super().__init__(input_data_shape, output_data_shape, config)

		tl_output = self.__build_RNN(self.Xs)
		self.build_flatten_layer(tl_output)

	def __build_RNN(self, Xs):
		def build_RNN_network(input_X, is_training=1):
			with tf.variable_scope('RNN'):
				network = tl.layers.FlattenLayer(input_X, name='flatten_layer_1')
				print('rnn input {}'.format(network.outputs.get_shape().as_list()))
				network = tl.layers.ReshapeLayer(network, shape=[-1, self.RNN_num_step, self.input_channel * self.input_vertical * self.input_horizontal], name='reshape_layer_1')
				network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_1')
				print('rnn input {}'.format(network.outputs.get_shape().as_list()))
				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=tf.nn.rnn_cell.LSTMCell,
					n_hidden=128,
					initializer=tf.random_uniform_initializer(-0.1, 0.1),
					n_steps=self.RNN_num_step,
					fw_initial_state=None,
					bw_initial_state=None,
					return_last=True,
					return_seq_2d=False,
					n_layer=3,
					dropout=(0.9, 0.9),
					name='layer_1')
				# print('rnn output {}'.format(network.outputs.get_shape().as_list()))
			return network

		tl_input = tl.layers.InputLayer(Xs, name='input_layer')
		RNN_tl_output = build_RNN_network(tl_input)
		return RNN_tl_output


class CNN_RNN(Multitask_Neural_Network):
	def __init__(self, input_data_shape, output_data_shape, config):
		# super().__init__(input_data_shape, output_data_shape, config)
		# self.norm = tf.placeholder(tf.bool, name='norm')

		# operation
		'''
		self.RNN_states_series, self.RNN_current_state = self._build_RNN_network_tf(network, self.keep_prob)
		RNN_last_output = tf.unpack(tf.transpose(self.RNN_states_series, [1, 0, 2]))  # a (batch_size, state_size) list with len num_step
		output_layer = tf.add(tf.matmul(RNN_last_output[-1], self.weights['output_layer']), self.bias['output_layer'])
		'''
		# network = tl.layers.DenseLayer(self.tl_RNN_output, n_units=200, act=tf.nn.relu, name='dense_layer')
		# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_1')
		self.__parse_config(config)
		self.build_MTL(input_data_shape, output_data_shape)
		tl_output = self.__build_CNN_RNN(self.Xs)
		self.build_flatten_layer(tl_output)

	def __build_CNN_RNN(self, Xs):
		def _build_Alex_CNN(input_X, is_training=1):
			with tf.variable_scope('CNN'):

				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				# print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.Conv2dLayer(
					network,
					act=tf.nn.relu,
					W_init=tf.truncated_normal_initializer(stddev=0.001),
					b_init=tf.constant_initializer(value=0.0),
					shape=[5, 5, 1, 32],
					strides=[1, 1, 1, 1],
					name='layer_1')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 3, 3, 1],
					strides=[1, 2, 2, 1],
					padding='SAME',
					pool=tf.nn.max_pool,
					name='pool_layer_1')
				network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=tf.nn.relu,
					W_init=tf.truncated_normal_initializer(stddev=0.01),
					b_init=tf.constant_initializer(value=0.0),
					shape=[5, 5, 32, 32],
					strides=[1, 1, 1, 1],
					name='layer_2')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 3, 3, 1],
					strides=[1, 2, 2, 1],
					padding='SAME',
					pool=tf.nn.avg_pool,
					name='pool_layer_2')
				network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_2')
				network = tl.layers.Conv2dLayer(
					network,
					act=tf.nn.relu,
					W_init=tf.truncated_normal_initializer(stddev=0.01),
					b_init=tf.constant_initializer(value=0.0),
					shape=[5, 5, 32, 64],
					strides=[1, 1, 1, 1],
					name='layer_3')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_3')
				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 3, 3, 1],
					strides=[1, 2, 2, 1],
					padding='SAME',
					pool=tf.nn.avg_pool,
					name='pool_layer_3')
			return network

		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				# print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_1')
				network_5x5 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_5x5_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_5x5_kernel_shape,
					strides=self.CNN_layer_1_5x5_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_5x5')

				network_5x5 = tl.layers.PoolLayer(
					network_5x5,
					ksize=self.CNN_layer_1_5x5_pooling_ksize,
					strides=self.CNN_layer_1_5x5_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_5x5_pooling,
					name='pool_layer_1_5x5')

				network_3x3 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_3x3_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_3x3_kernel_shape,
					strides=self.CNN_layer_1_3x3_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_3x3')

				network_3x3 = tl.layers.PoolLayer(
					network_3x3,
					ksize=self.CNN_layer_1_3x3_pooling_ksize,
					strides=self.CNN_layer_1_3x3_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_3x3_pooling,
					name='pool_layer_1_3x3')

				network = tl.layers.ConcatLayer(layer=[network_3x3, network_5x5], concat_dim=3, name='concate_layer_1')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_1_pooling_ksize,
					strides=self.CNN_layer_1_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_pooling,
					name='pool_layer_1')

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_2_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_2_kernel_shape,
					strides=self.CNN_layer_2_strides,
					padding='SAME',
					name='cnn_layer_2')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='pool_layer_2')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				return network

		def build_bi_RNN_network(input_X, is_training=1):
			def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unpack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state

			# print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('BI_RNN'):
				input_X = tl.layers.BatchNormLayer(input_X, is_train=is_training, name='batchnorm_layer_1')
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1')

				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=self.RNN_cell,
					cell_init_args=self.RNN_cell_init_args,
					n_hidden=self.RNN_hidden_node_size,
					initializer=self.RNN_initializer,
					n_steps=self.RNN_num_step,
					fw_initial_state=make_gaussan_state_initial('fw'),
					bw_initial_state=make_gaussan_state_initial('bw'),
					return_last=True,
					return_seq_2d=False,
					n_layer=self.RNN_num_layers,
					dropout=(self.keep_rate, self.keep_rate),
					name='layer_1')
				return network

		def build_RNN_network(input_X, is_training=1):
			print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('RNN'):
				input_X = tl.layers.BatchNormLayer(input_X, name='batchnorm_layer_1')
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1')
				network = tl.layers.RNNLayer(
					network,
					cell_fn=tf.nn.rnn_cell.GRUCell,
					# cell_init_args={'forget_bias': 0.0},
					n_hidden=self.RNN_hidden_node_size,
					initializer=tf.random_uniform_initializer(-0.1, 0.1),
					n_steps=self.RNN_num_step,
					initial_state=None,
					return_last=False,
					return_seq_2d=False,  # trigger with return_last is False. if True, return shape: (?, 200); if False, return shape: (?, 6, 200)
					name='layer_1')
				if is_training:
					network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				network = tl.layers.RNNLayer(
					network,
					cell_fn=tf.nn.rnn_cell.GRUCell,
					# cell_init_args={'forget_bias': 0.0},
					n_hidden=self.RNN_hidden_node_size,
					initializer=tf.random_uniform_initializer(-0.1, 0.1),
					n_steps=self.RNN_num_step,
					initial_state=None,
					return_last=False,
					return_seq_2d=False,  # trigger with return_last is False. if True, return shape: (?, 200); if False, return shape: (?, 6, 200)
					name='layer_2')
				if is_training:
					network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
				'''
				network = tl.layers.RNNLayer(
					network,
					cell_fn=tf.nn.rnn_cell.GRUCell,
					# cell_init_args={'forget_bias': 0.0},
					n_hidden=self.RNN_hidden_node_size,
					initializer=tf.random_uniform_initializer(-0.1, 0.1),
					n_steps=self.RNN_num_step,
					initial_state=None,
					return_last=False,
					return_seq_2d=False,  # trigger with return_last is False. if True, return shape: (?, 200); if False, return shape: (?, 6, 200)
					name='basic_lstm_layer_3')
				if is_training:
					network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_3')
				'''
				network = tl.layers.RNNLayer(
					network,
					cell_fn=tf.nn.rnn_cell.GRUCell,
					# cell_init_args={'forget_bias': 0.0},
					n_hidden=self.RNN_hidden_node_size,
					initializer=tf.random_uniform_initializer(-0.1, 0.1),
					n_steps=self.RNN_num_step,
					initial_state=None,
					return_last=True,
					return_seq_2d=False,  # trigger with return_last is False. if True, return shape: (?, 200); if False, return shape: (?, 6, 200)
					name='layer_3')
				if is_training:
					network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_3')
			'''
			network = tl.layers.DynamicRNNLayer(
				network,
				cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
				n_hidden=64,
				initializer=tf.random_uniform_initializer(-0.1, 0.1),
				n_steps=num_step,
				return_last=False,
				return_seq_2d=True,
				name='basic_lstm_layer_2')
			'''
			return network

		def build_inception_CNN_network(input_X, is_training=1):
			filter_map_1 = 16
			filter_map_2 = 16
			reduce1x1 = 10
			def create_inception(input_tl, filter_map_size, reduce_size, scope_name):
				input_shape = input_tl.outputs.get_shape().as_list()
				input_channel = input_shape[-1]
				with tf.variable_scope(scope_name):
					conv1_1x1_1 = tl.layers.Conv2dLayer(
						input_tl,
						act=tf.nn.relu,
						W_init=tf.contrib.layers.xavier_initializer_conv2d(),
						b_init=tf.constant_initializer(value=0.0),
						shape=[1, 1, input_channel, filter_map_size],
						strides=[1, 1, 1, 1],
						name='inception_1x1_1')
					conv1_1x1_1 = tl.layers.DropoutLayer(conv1_1x1_1, keep=self.keep_rate, name='drop_1')

					conv1_1x1_2 = tl.layers.Conv2dLayer(
						input_tl,
						act=tf.nn.relu,
						W_init=tf.contrib.layers.xavier_initializer_conv2d(),
						b_init=tf.constant_initializer(value=0.0),
						shape=[1, 1, input_channel, reduce_size],
						strides=[1, 1, 1, 1],
						name='inception_1x1_2')
					conv1_1x1_2 = tl.layers.DropoutLayer(conv1_1x1_2, keep=self.keep_rate, name='drop_2')

					conv1_1x1_3 = tl.layers.Conv2dLayer(
						input_tl,
						act=tf.nn.relu,
						W_init=tf.contrib.layers.xavier_initializer_conv2d(),
						b_init=tf.constant_initializer(value=0.0),
						shape=[1, 1, input_channel, reduce_size],
						strides=[1, 1, 1, 1],
						name='inception_1x1_3')
					conv1_1x1_3 = tl.layers.DropoutLayer(conv1_1x1_3, keep=self.keep_rate, name='drop_3')

					conv1_3x3 = tl.layers.Conv2dLayer(
						conv1_1x1_2,
						act=tf.nn.relu,
						W_init=tf.contrib.layers.xavier_initializer_conv2d(),
						b_init=tf.constant_initializer(value=0.0),
						shape=[3, 3, reduce_size, filter_map_size],
						strides=[1, 1, 1, 1],
						name='inception_3x3')
					conv1_3x3 = tl.layers.DropoutLayer(conv1_3x3, keep=self.keep_rate, name='drop_4')

					conv1_5x5 = tl.layers.Conv2dLayer(
						conv1_1x1_3,
						act=tf.nn.relu,
						W_init=tf.contrib.layers.xavier_initializer_conv2d(),
						b_init=tf.constant_initializer(value=0.0),
						shape=[5, 5, reduce_size, filter_map_size],
						strides=[1, 1, 1, 1],
						name='inception_5x5')
					conv1_5x5 = tl.layers.DropoutLayer(conv1_5x5, keep=self.keep_rate, name='drop_5')
					max_pool1 = tl.layers.PoolLayer(
						input_tl,
						ksize=[1, 3, 3, 1],
						strides=[1, 1, 1, 1],
						padding='SAME',
						pool=tf.nn.max_pool,
						name='inception_pool_layer_1')

					conv1_1x1_4 = tl.layers.Conv2dLayer(
						max_pool1,
						act=tf.nn.relu,
						W_init=tf.contrib.layers.xavier_initializer_conv2d(),
						b_init=tf.constant_initializer(value=0.0),
						shape=[1, 1, input_channel, filter_map_size],
						strides=[1, 1, 1, 1],
						name='inception1_1x1_4')
					conv1_1x1_4 = tl.layers.DropoutLayer(conv1_1x1_4, keep=self.keep_rate, name='drop_8')
					inception = tl.layers.ConcatLayer(layer=[conv1_1x1_1, conv1_3x3, conv1_5x5, conv1_1x1_4], concat_dim=3, name='concate_layer1')
					return inception

			with tf.variable_scope('CNN'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_1')
				inception = create_inception(network, filter_map_1, reduce1x1, 'inception_1')
				inception = tl.layers.DropoutLayer(inception, keep=self.keep_rate, name='drop_1')
				inception = tl.layers.PoolLayer(
					inception,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='SAME',
					pool=tf.nn.max_pool,
					name='pool_layer_1')
				network = tl.layers.Conv2dLayer(
					inception,
					act=tf.nn.relu,
					W_init=tf.contrib.layers.xavier_initializer_conv2d(),
					b_init=tf.constant_initializer(value=0.0),
					shape=[5, 5, 64, 64],
					strides=[1, 1, 1, 1],
					padding='SAME',
					name='cnn_layer_1')
				network = tl.layers.PoolLayer(
					network,
					ksize=[1, 3, 3, 1],
					strides=[1, 1, 1, 1],
					padding='SAME',
					pool=tf.nn.avf_pool,
					name='pool_layer_1')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				print('network output shape:{}'.format(inception.outputs.get_shape()))
				return inception

		with tf.variable_scope('CNN_RNN'):
			tl_CNN_output = build_CNN_network(Xs, is_training=False)

			cnn_flat = tl.layers.FlattenLayer(tl_CNN_output, name='CNN_flatten')
			network = tl.layers.BatchNormLayer(cnn_flat, is_train=False, name='batchnorm_layer_1')
			tl_RNN_output = build_bi_RNN_network(network, is_training=False)
			# print('CNN output:{}  rnn output:{}'.format(tl_CNN_output.outputs.get_shape().as_list(), tl_RNN_output.outputs.get_shape().as_list()))
			'''
			rnn_flat = tl.layers.FlattenLayer(tl_RNN_output, name='RNN_flatten')
			cnn_flat = tl.layers.ReshapeLayer(cnn_flat, [-1, self.input_temporal * cnn_flat.outputs.get_shape().as_list()[-1]])
			CNN_RNN_ouput_tl = tl.layers.ConcatLayer([cnn_flat, rnn_flat], concat_dim=1, name='CNN_RNN_concat')
			CNN_RNN_ouput_tl = tl.layers.DropoutLayer(CNN_RNN_ouput_tl, keep=0.9, name='dropout_layer')
			'''
			CNN_RNN_ouput_tl = tl_RNN_output

		return CNN_RNN_ouput_tl

	def _build_RNN_network_tf(self, input_X, keep_rate):
		def _get_states():
			state_per_layer_list = tf.unpack(self.RNN_init_state, axis=0)
			rnn_tuple_state = tuple(
				[tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(self.RNN_num_layers)])
			return rnn_tuple_state
		with tf.variable_scope('RNN'):
			input_X = input_X.outputs
			print('rnn network input shape :{}'.format(input_X.get_shape()))
			input_X = tf.reshape(input_X, [-1, self.RNN_num_step, input_X.get_shape().as_list()[-1]])
			print('rnn reshape input shape :{}'.format(input_X.get_shape()))
			RNN_cell = tf.nn.rnn_cell.BasicLSTMCell(self.RNN_hidden_node_size, state_is_tuple=True)
			RNN_cell = tf.nn.rnn_cell.DropoutWrapper(RNN_cell, output_keep_prob=keep_rate)
			RNN_cell = tf.nn.rnn_cell.MultiRNNCell([RNN_cell] * self.RNN_num_layers, state_is_tuple=True)

			status_tuple = _get_states()
			states_series, current_state = tf.nn.dynamic_rnn(RNN_cell, input_X, initial_state=status_tuple, time_major=False)

		return states_series, current_state

	def __parse_config(self, config):
		self.iter_epoch = config.iter_epoch
		self.batch_size = config.batch_size
		self.learning_rate = config.learning_rate
		self.weight_decay = config.weight_decay
		self.keep_rate = config.keep_rate
		self.RNN_num_layers = config.RNN_num_layers
		self.RNN_num_step = config.RNN_num_step
		self.RNN_hidden_node_size = config.RNN_hidden_node_size

		self.RNN_cell = self.parse_RNN_cell(config.RNN_cell)
		self.RNN_cell_init_args = config.RNN_cell_init_args
		self.RNN_init_state_noise_stddev = config.RNN_init_state_noise_stddev
		self.RNN_initializer = self.parse_initializer_method(config.RNN_initializer)

		self.CNN_layer_activation_fn = self.parse_activation(config.CNN_layer_activation_fn)
		self.CNN_layer_1_5x5_kernel_shape = config.CNN_layer_1_5x5_kernel_shape
		self.CNN_layer_1_5x5_kernel_strides = config.CNN_layer_1_5x5_kernel_strides
		self.CNN_layer_1_5x5_conv_Winit = self.parse_initializer_method(config.CNN_layer_1_5x5_conv_Winit)

		self.CNN_layer_1_5x5_pooling = self.parse_pooling(config.CNN_layer_1_5x5_pooling)
		self.CNN_layer_1_5x5_pooling_ksize = config.CNN_layer_1_5x5_pooling_ksize
		self.CNN_layer_1_5x5_pooling_strides = config.CNN_layer_1_5x5_pooling_strides

		self.CNN_layer_1_3x3_kernel_shape = config.CNN_layer_1_3x3_kernel_shape
		self.CNN_layer_1_3x3_kernel_strides = config.CNN_layer_1_3x3_kernel_strides
		self.CNN_layer_1_3x3_conv_Winit = self.parse_initializer_method(config.CNN_layer_1_3x3_conv_Winit)

		self.CNN_layer_1_3x3_pooling = self.parse_pooling(config.CNN_layer_1_3x3_pooling)
		self.CNN_layer_1_3x3_pooling_ksize = config.CNN_layer_1_3x3_pooling_ksize
		self.CNN_layer_1_3x3_pooling_strides = config.CNN_layer_1_3x3_pooling_strides

		self.CNN_layer_1_pooling = self.parse_pooling(config.CNN_layer_1_pooling)
		self.CNN_layer_1_pooling_ksize = config.CNN_layer_1_pooling_ksize
		self.CNN_layer_1_pooling_strides = config.CNN_layer_1_pooling_strides

		self.CNN_layer_2_kernel_shape = config.CNN_layer_2_kernel_shape
		self.CNN_layer_2_strides = config.CNN_layer_2_strides
		self.CNN_layer_2_conv_Winit = self.parse_initializer_method(config.CNN_layer_2_conv_Winit)

		self.CNN_layer_2_pooling_ksize = config.CNN_layer_2_pooling_ksize
		self.CNN_layer_2_pooling_strides = config.CNN_layer_2_pooling_strides
		self.CNN_layer_2_pooling = self.parse_pooling(config.CNN_layer_2_pooling)

		self.fully_connected_W_init = self.parse_initializer_method(config.fully_connected_W_init)
		self.fully_connected_units = config.fully_connected_units
		self.prediction_layer_1_W_init = self.parse_initializer_method(config.prediction_layer_1_W_init)
		self.prediction_layer_1_uints = config.prediction_layer_1_uints
		self.prediction_layer_2_W_init = self.parse_initializer_method(config.prediction_layer_2_W_init)
		self.prediction_layer_keep_rate = config.prediction_layer_keep_rate

		self.hyper_config = config


class CNN_RNN_2(Multitask_Neural_Network):
	def __init__(self, input_data_shape, output_data_shape, config):
		# super().__init__(input_data_shape, output_data_shape, config)
		self.__parse_config(config)
		self.build_MTL(input_data_shape, output_data_shape)
		tl_output = self.__build_CNN_RNN(self.Xs)
		self.build_flatten_layer(tl_output)

	def __build_CNN_RNN(self, Xs):
		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				# print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_kernel_shape,
					strides=self.CNN_layer_1_strides,
					padding='SAME',
					name='cnn_layer_1')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_1_pooling_ksize,
					strides=self.CNN_layer_1_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_pooling,
					name='pool_layer_1')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_2_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_2_kernel_shape,
					strides=self.CNN_layer_2_strides,
					padding='SAME',
					name='cnn_layer_2')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='pool_layer_2')

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_3_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_3_kernel_shape,
					strides=self.CNN_layer_3_strides,
					padding='SAME',
					name='cnn_layer_3')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_3_pooling_ksize,
					strides=self.CNN_layer_3_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_3_pooling,
					name='pool_layer_3')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_3')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				return network

		def build_bi_RNN_network(input_X, is_training=1):
			def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unpack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state

			# print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('BI_RNN'):
				input_X = tl.layers.BatchNormLayer(input_X, is_train=is_training, name='batchnorm_layer_1')
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1')

				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=self.RNN_cell,
					cell_init_args=self.RNN_cell_init_args,
					n_hidden=self.RNN_hidden_node_size,
					initializer=self.RNN_initializer,
					n_steps=self.RNN_num_step,
					fw_initial_state=make_gaussan_state_initial('fw'),
					bw_initial_state=make_gaussan_state_initial('bw'),
					return_last=True,
					return_seq_2d=False,
					n_layer=self.RNN_num_layers,
					dropout=(self.keep_rate, self.keep_rate),
					name='layer_1')
				return network
		with tf.variable_scope('CNN_RNN'):
			tl_CNN_output = build_CNN_network(Xs, is_training=False)

			cnn_flat = tl.layers.FlattenLayer(tl_CNN_output, name='CNN_flatten')
			network = tl.layers.BatchNormLayer(cnn_flat, is_train=False, name='batchnorm_layer_1')
			tl_RNN_output = build_bi_RNN_network(network, is_training=False)
			CNN_RNN_ouput_tl = tl_RNN_output

		return CNN_RNN_ouput_tl

	def __parse_config(self, config):
		self.iter_epoch = config.iter_epoch
		self.batch_size = config.batch_size
		self.learning_rate = config.learning_rate
		self.weight_decay = config.weight_decay
		self.keep_rate = config.keep_rate
		self.RNN_num_layers = config.RNN_num_layers
		self.RNN_num_step = config.RNN_num_step
		self.RNN_hidden_node_size = config.RNN_hidden_node_size

		self.RNN_cell = self.parse_RNN_cell(config.RNN_cell)
		self.RNN_cell_init_args = config.RNN_cell_init_args
		self.RNN_init_state_noise_stddev = config.RNN_init_state_noise_stddev
		self.RNN_initializer = self.parse_initializer_method(config.RNN_initializer)

		self.CNN_layer_activation_fn = self.parse_activation(config.CNN_layer_activation_fn)

		self.CNN_layer_1_kernel_shape = config.CNN_layer_1_kernel_shape
		self.CNN_layer_1_strides = config.CNN_layer_1_strides
		self.CNN_layer_1_conv_Winit = self.parse_initializer_method(config.CNN_layer_1_conv_Winit)

		self.CNN_layer_1_pooling = self.parse_pooling(config.CNN_layer_1_pooling)
		self.CNN_layer_1_pooling_ksize = config.CNN_layer_1_pooling_ksize
		self.CNN_layer_1_pooling_strides = config.CNN_layer_1_pooling_strides

		self.CNN_layer_2_kernel_shape = config.CNN_layer_2_kernel_shape
		self.CNN_layer_2_strides = config.CNN_layer_2_strides
		self.CNN_layer_2_conv_Winit = self.parse_initializer_method(config.CNN_layer_2_conv_Winit)

		self.CNN_layer_2_pooling_ksize = config.CNN_layer_2_pooling_ksize
		self.CNN_layer_2_pooling_strides = config.CNN_layer_2_pooling_strides
		self.CNN_layer_2_pooling = self.parse_pooling(config.CNN_layer_2_pooling)

		self.CNN_layer_3_kernel_shape = config.CNN_layer_3_kernel_shape
		self.CNN_layer_3_strides = config.CNN_layer_3_strides
		self.CNN_layer_3_conv_Winit = self.parse_initializer_method(config.CNN_layer_3_conv_Winit)

		self.CNN_layer_3_pooling_ksize = config.CNN_layer_3_pooling_ksize
		self.CNN_layer_3_pooling_strides = config.CNN_layer_3_pooling_strides
		self.CNN_layer_3_pooling = self.parse_pooling(config.CNN_layer_3_pooling)

		self.fully_connected_W_init = self.parse_initializer_method(config.fully_connected_W_init)
		self.fully_connected_units = config.fully_connected_units
		self.prediction_layer_1_W_init = self.parse_initializer_method(config.prediction_layer_1_W_init)
		self.prediction_layer_1_uints = config.prediction_layer_1_uints
		self.prediction_layer_2_W_init = self.parse_initializer_method(config.prediction_layer_2_W_init)
		self.prediction_layer_keep_rate = config.prediction_layer_keep_rate

		self.hyper_config = config


class concurrent_CNN_RNN(Multitask_Neural_Network):
	def __init__(self, input_data_shape, output_data_shape, config):
		# super().__init__(input_data_shape, output_data_shape, config)
		self.__parse_config(config)
		self.build_MTL(input_data_shape, output_data_shape)
		with tf.variable_scope('concurrent_CNN_RNN'):
			tl_RNN_ouput = self.__build_RNN(self.Xs)
			tl_CNN_output = self.__build_CNN(self.Xs, is_training=False)
			# print(tl_RNN_ouput.outputs.get_shape().as_list())
			# print(tl_CNN_output.outputs.get_shape().as_list())
			CNN_output_shape = tl_CNN_output.outputs.get_shape().as_list()
			tl_RNN_output = tl.layers.ReshapeLayer(tl_RNN_ouput, shape=[self.batch_size, -1], name='RNN_reshape_layer')
			tl_CNN_output = tl.layers.ReshapeLayer(tl_CNN_output, shape=[self.batch_size, CNN_output_shape[1] * CNN_output_shape[2] * CNN_output_shape[3]], name='CNN_reshape_layer')
			# print(tl_RNN_ouput.outputs.get_shape().as_list())
			# print(tl_CNN_output.outputs.get_shape().as_list())
			tl_output = tl.layers.ConcatLayer(layer=[tl_CNN_output, tl_RNN_output], concat_dim=1, name='concate_layer1')

		self.build_flatten_layer(tl_output)
		# self.build_flatten_layer(tl_output)

	def __build_CNN(self, Xs, is_training=1):
		with tf.variable_scope('CNN'):
			CNN_input = tf.reshape(Xs, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
			network = tl.layers.InputLayer(CNN_input, name='CNN_input_layer')
			network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_1')
			network_5x5 = tl.layers.Conv2dLayer(
				network,
				act=self.CNN_layer_activation_fn,
				W_init=self.CNN_layer_1_5x5_conv_Winit,
				b_init=tf.constant_initializer(value=0.01),
				shape=self.CNN_layer_1_5x5_kernel_shape,
				strides=self.CNN_layer_1_5x5_kernel_strides,
				padding='SAME',
				name='cnn_layer_1_5x5')

			network_5x5 = tl.layers.PoolLayer(
				network_5x5,
				ksize=self.CNN_layer_1_5x5_pooling_ksize,
				strides=self.CNN_layer_1_5x5_pooling_strides,
				padding='SAME',
				pool=self.CNN_layer_1_5x5_pooling,
				name='pool_layer_1_5x5')

			network_3x3 = tl.layers.Conv2dLayer(
				network,
				act=self.CNN_layer_activation_fn,
				W_init=self.CNN_layer_1_3x3_conv_Winit,
				b_init=tf.constant_initializer(value=0.01),
				shape=self.CNN_layer_1_3x3_kernel_shape,
				strides=self.CNN_layer_1_3x3_kernel_strides,
				padding='SAME',
				name='cnn_layer_1_3x3')

			network_3x3 = tl.layers.PoolLayer(
				network_3x3,
				ksize=self.CNN_layer_1_3x3_pooling_ksize,
				strides=self.CNN_layer_1_3x3_pooling_strides,
				padding='SAME',
				pool=self.CNN_layer_1_3x3_pooling,
				name='pool_layer_1_3x3')

			network = tl.layers.ConcatLayer(layer=[network_3x3, network_5x5], concat_dim=3, name='concate_layer_1')

			network = tl.layers.PoolLayer(
				network,
				ksize=self.CNN_layer_1_pooling_ksize,
				strides=self.CNN_layer_1_pooling_strides,
				padding='SAME',
				pool=self.CNN_layer_1_pooling,
				name='pool_layer_1')

			network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
			network = tl.layers.Conv2dLayer(
				network,
				act=self.CNN_layer_activation_fn,
				W_init=self.CNN_layer_2_conv_Winit,
				b_init=tf.constant_initializer(value=0.01),
				shape=self.CNN_layer_2_kernel_shape,
				strides=self.CNN_layer_2_strides,
				padding='SAME',
				name='cnn_layer_2')

			network = tl.layers.PoolLayer(
				network,
				ksize=self.CNN_layer_2_pooling_ksize,
				strides=self.CNN_layer_2_pooling_strides,
				padding='SAME',
				pool=self.CNN_layer_2_pooling,
				name='pool_layer_2')
			network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
			# network = tl.layers.FlattenLayer(network, name='flatten_layer')

			# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
			# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
			# print('network output shape:{}'.format(network.outputs.get_shape()))
		return network

	def __build_RNN(self, Xs):
		def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unpack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state
		with tf.variable_scope('BI_RNN'):
			# print(Xs.get_shape().as_list())
			input_slice = tf.slice(Xs, [0, 0, 7, 7, 0], [-1, -1, 1, 1, -1])
			# print(input_slice.get_shape().as_list())
			RNN_input = tf.reshape(input_slice, [-1, self.input_temporal, self.input_channel])
	
			tl_RNN_input = tl.layers.InputLayer(RNN_input, name='RNN_input_layer')
			network = tl.layers.BiRNNLayer(
				tl_RNN_input,
				cell_fn=self.RNN_cell,
				cell_init_args=self.RNN_cell_init_args,
				n_hidden=self.RNN_hidden_node_size,
				initializer=self.RNN_initializer,
				n_steps=self.RNN_num_step,
				fw_initial_state=make_gaussan_state_initial('fw'),
				bw_initial_state=make_gaussan_state_initial('bw'),
				return_last=True,
				return_seq_2d=False,
				n_layer=self.RNN_num_layers,
				dropout=(self.keep_rate, self.keep_rate),
				name='layer_1')
			return network

	def __parse_config(self, config):
		self.iter_epoch = config.iter_epoch
		self.batch_size = config.batch_size
		self.learning_rate = config.learning_rate
		self.weight_decay = config.weight_decay
		self.keep_rate = config.keep_rate
		self.RNN_num_layers = config.RNN_num_layers
		self.RNN_num_step = config.RNN_num_step
		self.RNN_hidden_node_size = config.RNN_hidden_node_size

		self.RNN_cell = self.parse_RNN_cell(config.RNN_cell)
		self.RNN_cell_init_args = config.RNN_cell_init_args
		self.RNN_init_state_noise_stddev = config.RNN_init_state_noise_stddev
		self.RNN_initializer = self.parse_initializer_method(config.RNN_initializer)

		self.CNN_layer_activation_fn = self.parse_activation(config.CNN_layer_activation_fn)
		self.CNN_layer_1_5x5_kernel_shape = config.CNN_layer_1_5x5_kernel_shape
		self.CNN_layer_1_5x5_kernel_strides = config.CNN_layer_1_5x5_kernel_strides
		self.CNN_layer_1_5x5_conv_Winit = self.parse_initializer_method(config.CNN_layer_1_5x5_conv_Winit)

		self.CNN_layer_1_5x5_pooling = self.parse_pooling(config.CNN_layer_1_5x5_pooling)
		self.CNN_layer_1_5x5_pooling_ksize = config.CNN_layer_1_5x5_pooling_ksize
		self.CNN_layer_1_5x5_pooling_strides = config.CNN_layer_1_5x5_pooling_strides

		self.CNN_layer_1_3x3_kernel_shape = config.CNN_layer_1_3x3_kernel_shape
		self.CNN_layer_1_3x3_kernel_strides = config.CNN_layer_1_3x3_kernel_strides
		self.CNN_layer_1_3x3_conv_Winit = self.parse_initializer_method(config.CNN_layer_1_3x3_conv_Winit)

		self.CNN_layer_1_3x3_pooling = self.parse_pooling(config.CNN_layer_1_3x3_pooling)
		self.CNN_layer_1_3x3_pooling_ksize = config.CNN_layer_1_3x3_pooling_ksize
		self.CNN_layer_1_3x3_pooling_strides = config.CNN_layer_1_3x3_pooling_strides

		self.CNN_layer_1_pooling = self.parse_pooling(config.CNN_layer_1_pooling)
		self.CNN_layer_1_pooling_ksize = config.CNN_layer_1_pooling_ksize
		self.CNN_layer_1_pooling_strides = config.CNN_layer_1_pooling_strides

		self.CNN_layer_2_kernel_shape = config.CNN_layer_2_kernel_shape
		self.CNN_layer_2_strides = config.CNN_layer_2_strides
		self.CNN_layer_2_conv_Winit = self.parse_initializer_method(config.CNN_layer_2_conv_Winit)

		self.CNN_layer_2_pooling_ksize = config.CNN_layer_2_pooling_ksize
		self.CNN_layer_2_pooling_strides = config.CNN_layer_2_pooling_strides
		self.CNN_layer_2_pooling = self.parse_pooling(config.CNN_layer_2_pooling)

		self.fully_connected_W_init = self.parse_initializer_method(config.fully_connected_W_init)
		self.fully_connected_units = config.fully_connected_units
		self.prediction_layer_1_W_init = self.parse_initializer_method(config.prediction_layer_1_W_init)
		self.prediction_layer_1_uints = config.prediction_layer_1_uints
		self.prediction_layer_2_W_init = self.parse_initializer_method(config.prediction_layer_2_W_init)
		self.prediction_layer_keep_rate = config.prediction_layer_keep_rate
		self.hyper_config = config


