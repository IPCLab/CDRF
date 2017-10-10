import numpy as np
import os
import data_utility as du


class Prepare_Task_Data:

	def __init__(self, target_path):
		self.target_path = target_path
		self.root_dir = '/home/mldp/ML_with_bigdata'

	def _prepare_data(self, grid_limit, task_name):
		def load_and_save(file_dir, target_path):
			filelist = du.list_all_input_file(file_dir)
			filelist.sort()
			for i, filename in enumerate(filelist):
				file_path = os.path.join(file_dir, filename)
				data_array = du.load_array(file_path)
				data_array = data_array[:, :, grid_limit[0][0]: grid_limit[0][1], grid_limit[1][0]: grid_limit[1][1], (0, 1, -1)]
				print('saving array shape:', data_array.shape)
				du.save_array(data_array, os.path.join(target_path, task_name + '_' + str(i)))

		x_target_path = os.path.join(self.target_path, task_name, 'X')
		y_target_path = os.path.join(self.target_path, task_name, 'Y')
		if not os.path.exists(x_target_path):
			os.makedirs(x_target_path)
		if not os.path.exists(y_target_path):
			os.makedirs(y_target_path)

		# prepare x
		file_dir = os.path.join(self.root_dir, 'npy', task_name, 'X')
		load_and_save(file_dir, x_target_path)
		# prepare y
		file_dir = os.path.join(self.root_dir, 'npy', task_name, 'Y')
		load_and_save(file_dir, y_target_path)

	def _get_X_and_Y(self, task_name):
		def load_data(file_dir):
			file_list = du.list_all_input_file(file_dir)
			file_list.sort()
			array_list = []

			for filename in file_list:
				array_list.append(du.load_array(os.path.join(file_dir, filename)))
			data_array = np.concatenate(array_list, axis=0)
			return data_array

		x_dir = os.path.join(self.target_path, task_name, 'X')
		y_dir = os.path.join(self.target_path, task_name, 'Y')
		return load_data(x_dir), load_data(y_dir)

	def Task_max(self, grid_limit=[(45, 60), (45, 60)], generate_data=False):
		task_name = 'hour_max'
		if generate_data:
			self._prepare_data(grid_limit, task_name)
		X_array, Y_array = self._get_X_and_Y(task_name)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def Task_min(self, grid_limit=[(45, 60), (45, 60)], generate_data=False):
		task_name = 'hour_min'
		if generate_data:
			self._prepare_data(grid_limit, task_name)
		X_array, Y_array = self._get_X_and_Y(task_name)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes
		return X_array, Y_array

	def Task_avg(self, grid_limit=[(45, 60), (45, 60)], generate_data=False):
		task_name = 'hour_avg'
		if generate_data:
			self._prepare_data(grid_limit, task_name)
		X_array, Y_array = self._get_X_and_Y(task_name)
		X_array = X_array[0: -1]  # important!!
		Y_array = Y_array[1:]  # important!! Y should shift 10 minutes

		return X_array, Y_array

	def Task_max_min_avg(self, grid_limit=[(45, 60), (45, 60)], generate_data=False):
		# print(grid_limit)
		task_name = 'hour_min_avg_max'
		if generate_data:
			x_target_path = os.path.join(self.target_path, task_name, 'X')
			y_target_path = os.path.join(self.target_path, task_name, 'Y')
			if not os.path.exists(x_target_path):
				os.makedirs(x_target_path)
			if not os.path.exists(y_target_path):
				os.makedirs(y_target_path)
			X, max_Y = self.Task_max(grid_limit, generate_data)
			_, min_Y = self.Task_min(grid_limit, generate_data)
			_, avg_Y = self.Task_avg(grid_limit, generate_data)
			min_avg_max_Y = np.zeros([max_Y.shape[0], max_Y.shape[1], max_Y.shape[2], max_Y.shape[3], 5])  # grid_id timestamp, min, avg, max

			for i in range(max_Y.shape[0]):
				for j in range(max_Y.shape[1]):
					for row in range(max_Y.shape[2]):
						for col in range(max_Y.shape[3]):
							# print('min:{} avg:{} max:{}'.format(min_Y[i, j, row, col, 0], avg_Y[i, j, row, col, 0], max_Y[i, j, row, col, 0]))
							min_avg_max_Y[i, j, row, col, 0] = min_Y[i, j, row, col, 0]  # grid_id
							min_avg_max_Y[i, j, row, col, 1] = min_Y[i, j, row, col, 1]  # timesatemp

							min_avg_max_Y[i, j, row, col, 2] = min_Y[i, j, row, col, -1]  # internet traffic
							min_avg_max_Y[i, j, row, col, 3] = avg_Y[i, j, row, col, -1]  # internet traffic
							min_avg_max_Y[i, j, row, col, 4] = max_Y[i, j, row, col, -1]  # internet traffic
			du.save_array(X, os.path.join(x_target_path, 'min_avg_max_X'))
			du.save_array(min_avg_max_Y, os.path.join(y_target_path, 'min_avg_max_Y'))
			return X, min_avg_max_Y
		else:
			return self._get_X_and_Y(task_name)


if __name__ == '__main__':
	TK = Prepare_Task_Data('./npy/final')
	X_array, Y_array = TK.Task_max(grid_limit=[(45, 60), (45, 60)], generate_data=True)
	print(X_array.shape)
