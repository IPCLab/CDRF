import matplotlib.pyplot as plt
import utility
import numpy as np
import json


def plot_predict_vs_real(info, real, predict, title_name, interval=6):
	def get_xlabel(timestamps):
		xlabel_list = []
		for timestamp in timestamps:
			datetime = utility.set_time_zone(timestamp)
			xlabel_list.append(utility.date_time_covert_to_str(datetime))
		return xlabel_list

	x_len = info.shape[0]
	xlabel_list = get_xlabel(info[:, 1])
	fig, ax = plt.subplots(1, 1, figsize=(10, 4))
	ax.set_xlabel('Time sequence')
	ax.set_ylabel('Number of CDRs')

	ax.plot(range(x_len), real, label='Real', color='k')
	ax.plot(range(x_len), predict, label='Predictoin', color='r', linestyle='--')
	# plt.xticks(list(range(0, x_len, 12)), xlabel_list[:x_len:2], rotation=45)
	# plt.set_xticklabels()
	ax.set_xticks(list(range(0, x_len, interval)))
	ax.set_xticklabels(xlabel_list[0:x_len:interval], rotation=45)
	ax.set_title(title_name + ' grid id ' + str(int(info[0, 0])))
	ax.grid()
	ax.legend()
	return fig


def report_loss_accu(info, real, predict, file_path):
	real = np.transpose(real, (2, 3, 0, 1, 4))
	predict = np.transpose(predict, (2, 3, 0, 1, 4))
	info = np.transpose(info, (2, 3, 0, 1, 4))
	report_dict = {}

	def report_(real, predict):
		MAPE = utility.MAPE_loss(real, predict)
		AE = utility.AE_loss(real, predict)
		RMSE = utility.RMSE_loss(real, predict)
		Accu = 1 - MAPE if MAPE else None
		report_dict = {
			'MAPE': MAPE,
			'AE': AE,
			'RMSE': RMSE,
			'Accuracy': Accu
		}
		return report_dict

	def each_task(real, predict):
		min_report = report_(real[:, :, 0], predict[:, :, 0])
		avg_report = report_(real[:, :, 1], predict[:, :, 1])
		max_report = report_(real[:, :, 2], predict[:, :, 2])
		report_dict = {
			'task_min': min_report,
			'task_avg': avg_report,
			'task_max': max_report
		}

		return report_dict

	for row in range(info.shape[0]):
		for col in range(info.shape[1]):
			grid_id = info[row, col, 0, 0, 0]
			each_grid_report = each_task(real[row, col], predict[row, col])
			report_dict[str(int(grid_id))] = each_grid_report

	total_min = report_(real[:, :, :, :, 0], predict[:, :, :, :, 0])
	total_avg = report_(real[:, :, :, :, 1], predict[:, :, :, :, 1])
	total_max = report_(real[:, :, :, :, 2], predict[:, :, :, :, 2])
	total_report = {
		'task_min': total_min,
		'task_avg': total_avg,
		'task_max': total_max
	}
	report_dict['total'] = total_report

	with open(file_path, 'w') as outfile:
		json_string = json.dumps(report_dict, sort_keys=True, indent=4)
		outfile.write(json_string)

	return report_dict
