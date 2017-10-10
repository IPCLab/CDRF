import numpy as np
import os



def list_all_input_file(input_dir):
	onlyfile = [f for f in os.listdir(input_dir) if (os.path.isfile(
		os.path.join(input_dir, f)) and os.path.splitext(f)[1] == ".npy")]
	return onlyfile


def save_array(x_array, out_file):
	print('saving file to {}...'.format(out_file))
	np.save(out_file, x_array, allow_pickle=True)


def load_array(input_file):
	print('loading file from {}...'.format(input_file))
	X = np.load(input_file)
	return X


def get_one_hour_min(input_array):
	output_shape = [
		input_array.shape[0],
		1,
		input_array.shape[2],
		input_array.shape[3]]
	output_array = np.zeros(output_shape, dtype=np.float32)
	for i in range(input_array.shape[0]):
		for row in range(input_array.shape[2]):
			for col in range(input_array.shape[3]):
				min_value = np.amin(input_array[i, :, row, col])
				# print(input_array[i, :, row, col], ' min:', min_value)
				output_array[i, 0, row, col] = min_value

	# print('output array shape {}'.format(output_array.shape))
	return output_array


def get_one_hour_max(input_array):
	# print('input array shape {}'.format(input_array.shape))
	output_shape = [
		input_array.shape[0],
		1,
		input_array.shape[2],
		input_array.shape[3]]
	output_array = np.zeros(output_shape, dtype=np.float32)
	for i in range(input_array.shape[0]):
		for row in range(input_array.shape[2]):
			for col in range(input_array.shape[3]):
				max_value = np.amax(input_array[i, :, row, col])
				# print('record {} row {} col {} max {}'.format(i, row, col, max_value))
				output_array[i, 0, row, col] = max_value

	# print('output array shape {}'.format(output_array.shape))
	return output_array


def get_one_hour_average(input_array):
	print('input array shape {}'.format(input_array.shape))
	output_shape = [
		input_array.shape[0],
		1,
		input_array.shape[2],
		input_array.shape[3]]
	output_array = np.zeros(output_shape, dtype=np.float32)
	average_value_np = np.mean(input_array, axis=1)
	print(average_value_np.shape)
	for i in range(input_array.shape[0]):
		for row in range(input_array.shape[2]):
			for col in range(input_array.shape[3]):
				average_value = np.mean(input_array[i, :, row, col])
				output_array[i, 0, row, col] = average_value
	print('output_array shape {}'.format(output_array.shape))
	return output_array


def load_data_hour_average(input_dir, filelist):
	x_target_path = './npy/hour_avg/X/'
	y_target_path = './npy/hour_avg/Y/'
	if not os.path.exists(x_target_path):
		os.makedirs(x_target_path)
	if not os.path.exists(y_target_path):
		os.makedirs(y_target_path)

	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float32)

	def split_data(data_array):
		split_block_size = 6
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		split_block_num = int(data_array_depth / split_block_size)
		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)

		new_data_array = np.stack(split_data_list, axis=0)
		print('new_data_array shape {}'.format(new_data_array.shape))
		return new_data_array

	def one_hour_avg(input_array):
		grid_id = input_array[:, 0:1, :, :, 0]
		timestamp = input_array[:, 0:1, :, :, 1]
		call_in = get_one_hour_average(input_array[:, :, :, :, 2])
		call_out = get_one_hour_average(input_array[:, :, :, :, 3])
		sms_in = get_one_hour_average(input_array[:, :, :, :, 4])
		sms_out = get_one_hour_average(input_array[:, :, :, :, 5])
		internet = get_one_hour_average(input_array[:, :, :, :, 6])

		new_array_list = [grid_id, timestamp, call_in, call_out, sms_in, sms_out, internet]
		new_array = np.stack(new_array_list, axis=-1)

		print('new_array shape {}'.format(new_array.shape))
		return new_array

	month, _ = os.path.split(input_dir)
	month = month.split('/')[-2]
	data_array_list = []
	data_group_para = 10
	data_divide_amount = (len(filelist) // data_group_para) + 1
	for i, file_name in enumerate(filelist):
		data_array_list.append(load_array(input_dir + file_name))
		if i % data_group_para == 0 and i != 0:
			index = i // data_group_para
			data_array = np.concatenate(data_array_list, axis=0)
			data_array_list = []
			X = split_data(data_array)
			Y = one_hour_avg(X)
			save_array(X, x_target_path + month + '_hour_avg_X_' + str(index))
			save_array(Y, y_target_path + month + '_hour_avg_Y_' + str(index))
	# remainder
	data_array = np.concatenate(data_array_list, axis=0)
	del data_array_list
	X = split_data(data_array)
	Y = one_hour_avg(X)
	save_array(X, x_target_path + month + '_hour_avg_X_' + str(data_divide_amount))
	save_array(Y, y_target_path + month + '_hour_avg_Y_' + str(data_divide_amount))


def load_data_hour_min(input_dir, filelist):
	x_target_path = './npy/hour_min/X/'
	y_target_path = './npy/hour_min/Y/'
	if not os.path.exists(x_target_path):
		os.makedirs(x_target_path)
	if not os.path.exists(y_target_path):
		os.makedirs(y_target_path)

	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float32)

	def split_data(data_array):
		split_block_size = 6
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		split_block_num = int(data_array_depth / split_block_size)
		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)

		new_data_array = np.stack(split_data_list, axis=0)
		print('new_data_array shape {}'.format(new_data_array.shape))
		return new_data_array

	def one_hour_min(input_array):
		grid_id = input_array[:, 0:1, :, :, 0]
		timestamp = input_array[:, 0:1, :, :, 1]
		call_in = get_one_hour_min(input_array[:, :, :, :, 2])
		call_out = get_one_hour_min(input_array[:, :, :, :, 3])
		sms_in = get_one_hour_min(input_array[:, :, :, :, 4])
		sms_out = get_one_hour_min(input_array[:, :, :, :, 5])
		internet = get_one_hour_min(input_array[:, :, :, :, 6])

		new_array_list = [grid_id, timestamp, call_in, call_out, sms_in, sms_out, internet]
		new_array = np.stack(new_array_list, axis=-1)

		print('new_array shape {}'.format(new_array.shape))
		return new_array

	month, _ = os.path.split(input_dir)
	month = month.split('/')[-2]
	data_array_list = []
	data_group_para = 10
	data_divide_amount = (len(filelist) // data_group_para) + 1
	for i, file_name in enumerate(filelist):
		data_array_list.append(load_array(input_dir + file_name))
		if i % data_group_para == 0 and i != 0:
			index = i // data_group_para
			data_array = np.concatenate(data_array_list, axis=0)
			data_array_list = []
			X = split_data(data_array)
			Y = one_hour_min(X)
			save_array(X, x_target_path + month + '_hour_min_X_' + str(index))
			save_array(Y, y_target_path + month + '_hour_min_Y_' + str(index))
	# remainder
	data_array = np.concatenate(data_array_list, axis=0)
	del data_array_list
	X = split_data(data_array)
	Y = one_hour_min(X)
	save_array(X, x_target_path + month + '_hour_min_X_' + str(data_divide_amount))
	save_array(Y, y_target_path + month + '_hour_min_Y_' + str(data_divide_amount))


def load_data_hour_max(input_dir, filelist):
	x_target_path = './npy/hour_max/X/'
	y_target_path = './npy/hour_max/Y/'
	if not os.path.exists(x_target_path):
		os.makedirs(x_target_path)
	if not os.path.exists(y_target_path):
		os.makedirs(y_target_path)

	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float32)

	def split_data(data_array):
		split_block_size = 6
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		split_block_num = int(data_array_depth / split_block_size)
		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)

		new_data_array = np.stack(split_data_list, axis=0)
		print('new_data_array shape {}'.format(new_data_array.shape))
		return new_data_array

	def one_hour_max(input_array):
		grid_id = input_array[:, 0:1, :, :, 0]
		timestamp = input_array[:, 0:1, :, :, 1]
		call_in = get_one_hour_max(input_array[:, :, :, :, 2])
		call_out = get_one_hour_max(input_array[:, :, :, :, 3])
		sms_in = get_one_hour_max(input_array[:, :, :, :, 4])
		sms_out = get_one_hour_max(input_array[:, :, :, :, 5])
		internet = get_one_hour_max(input_array[:, :, :, :, 6])

		new_array_list = [grid_id, timestamp, call_in, call_out, sms_in, sms_out, internet]
		new_array = np.stack(new_array_list, axis=-1)

		print('new_array shape {}'.format(new_array.shape))
		return new_array

	month, _ = os.path.split(input_dir)
	month = month.split('/')[-2]
	data_array_list = []
	data_group_para = 10
	data_divide_amount = (len(filelist) // data_group_para) + 1
	for i, file_name in enumerate(filelist):
		data_array_list.append(load_array(input_dir + file_name))
		if i % data_group_para == 0 and i != 0:
			index = i // data_group_para
			data_array = np.concatenate(data_array_list, axis=0)
			data_array_list = []
			X = split_data(data_array)
			Y = one_hour_max(X)
			save_array(X, x_target_path + month + '_hour_max_X_' + str(index))
			save_array(Y, y_target_path + month + '_hour_max_Y_' + str(index))
	# remainder
	data_array = np.concatenate(data_array_list, axis=0)
	del data_array_list
	X = split_data(data_array)
	Y = one_hour_max(X)
	save_array(X, x_target_path + month + '_hour_max_X_' + str(data_divide_amount))
	save_array(Y, y_target_path + month + '_hour_max_Y_' + str(data_divide_amount))


def load_data_format_roll_10mins(input_dir, filelist):
	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float32)

	def split_data(data_array):
		split_block_size = 6
		data_array_depth = data_array.shape[0]
		new_data_length = data_array_depth - split_block_size

		# remainder = data_array_depth % split_block_size
		split_data_X_list = []
		split_data_Y_list = []
		for index in range(new_data_length):
			split_data_X = data_array[index: index + split_block_size]
			split_data_X = np.stack(split_data_X, axis=0)
			# print('split_data_X shape:{}'.format(split_data_X.shape))
			split_data_Y = data_array[index + split_block_size]
			split_data_X_list.append(split_data_X)
			split_data_Y_list.append(split_data_Y)
		X = np.stack(split_data_X_list, axis=0)
		del split_data_X_list
		Y = np.stack(split_data_Y_list, axis=0)
		Y = np.expand_dims(Y, axis=1)
		del split_data_Y_list
		print('x shape {} y shape {}'.format(X.shape, Y.shape))
		return X, Y

	month, _ = os.path.split(input_dir)
	month = month.split('/')[-2]
	data_array_list = []
	data_group_para = 10
	data_divide_amount = (len(filelist) // data_group_para) + 1
	for i, file_name in enumerate(filelist):
		data_array_list.append(load_array(input_dir + file_name))
		if i % data_group_para == 0 and i != 0:
			index = i // data_group_para
			data_array = np.concatenate(data_array_list, axis=0)
			data_array_list = []
			X, Y = split_data(data_array)
			save_array(X, './npy/npy_roll/X/' + month + '_roll_X_' + str(index))
			save_array(Y, './npy/npy_roll/Y/' + month + '_roll_y_' + str(index))
	# remainder
	data_array = np.concatenate(data_array_list, axis=0)
	del data_array_list
	X, Y = split_data(data_array)
	save_array(X, './npy/npy_roll/X/' + month + '_roll_X_' + str(data_divide_amount))
	save_array(Y, './npy/npy_roll/Y/' + month + '_roll_y_' + str(data_divide_amount))


def load_data_format(input_dir, filelist):
	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float32)

	def split_array(data_array):
		# print('data_array shape :', data_array.shape)
		split_block_size = 6  # one hour
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		split_block_num = int(data_array_depth / split_block_size)

		# new_data_array_size = [split_block_num,data_array.shape[1:]]
		# print('new_data_array_size:',new_data_array_size)

		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)
		new_data_array = np.stack(split_data_list, axis=0)
		# print('new_data_array shape:', new_data_array.shape)

		return new_data_array

	def shift_data(data_array):
		'''
			generate more data
		'''
		array_list = []
		for shift_index in [0, 2, 4, 5]:
			shift_array = data_array[shift_index:]
			array_list.append(split_array(shift_array))

		return array_list

	def array_concatenate(x, y):  # for reduce
		return np.concatenate((x, y), axis=0)

	month, _ = os.path.split(input_dir)
	month = month.split('/')[-2]

	data_array_list = []
	for file_name in filelist:
		data_array_list.append(load_array(input_dir + file_name))
	data_array = np.concatenate(data_array_list, axis=0)
	del data_array_list
	shift_data_array_list = shift_data(data_array)
	for i, array in enumerate(shift_data_array_list):
		print('data format shape:', array.shape)
		save_array(array, './npy/' + month + '_' + str(i))  # saving all shift array





		