from datetime import datetime, tzinfo, timedelta
from time import sleep, time
import calendar
import numpy as np
import pytz
import pickle
import os
input_dir_list = ["/home/mldp/big_data/openbigdata/milano/SMS/11/",
                  "/home/mldp/big_data/openbigdata/milano/SMS/12/"]
#input_file = "sms-call-internet-mi-2013-11-01.txt"
#ouput_dir = "/home/mldp/big_data/openbigdata/milano/SMS/11/data_preproccessing/"
#ouput_file = "output_sms-call-internet-mi-2013-11-01"


UTC_timezone = pytz.timezone('UTC')
Mi_timezone = pytz.timezone('Europe/Rome')


def list_all_input_file(input_dir):

    onlyfile = [f for f in os.listdir(input_dir) if (os.path.isfile(
        os.path.join(input_dir, f)) and os.path.splitext(f)[1] == ".txt")]
    return onlyfile


def save_preprecessing_data(Mi_data):
    with open(ouput_dir + ouput_file, 'w') as wf:
        for i, each_line in enumerate(Mi_data['timestamp']):
            wf.write(each_line, Mi_data['square_id_1'], Mi_data[
                     'square_id_2'], Mi_data['interaction'])


def combine_data(Mi_data, Mi_data_proceesed):
    start_time = set_time_zone(Mi_data['timestamp'][0])
    square_index = Mi_data['square_id'][0]
    time_interval = 10
    end_time = start_time + timedelta(minutes=time_interval)

    temp_activity = 0
    sms_in_activity = 0
    sms_out_activity = 0
    call_in_activity = 0
    call_out_activity = 0
    # Mi_data_proceesed={}
    for i, each_line in enumerate(Mi_data['timestamp']):
        timestamp = Mi_data['timestamp'][i]
        date_time = set_time_zone(timestamp)
        square_id = int(Mi_data['square_id'][i])
        sms_in_activity = float(Mi_data['sms_in_activity'][i])
        sms_out_activity = float(Mi_data['sms_out_activity'][i])
        call_in_activity = float(Mi_data['call_in_activity'][i])
        call_out_activity = float(Mi_data['call_out_activity'][i])
        internat_traffic_activity = float(
            Mi_data['internat_traffic_activity'][i])

        temp_activity += internat_traffic_activity
        sms_in_activity += sms_in_activity
        sms_out_activity += sms_out_activity
        call_in_activity += call_in_activity
        call_out_activity += call_out_activity

        if square_index != square_id:
            start_time = date_time
            end_time = start_time + timedelta(minutes=time_interval)
            square_index = square_id

        if end_time <= date_time + timedelta(minutes=10):
            end_time_str = date_time_covert_to_str(end_time)
            #end_timestamp = mktime(end_time.timetuple())
            end_timestamp = end_time.astimezone(UTC_timezone)
            end_timestamp = calendar.timegm(end_timestamp.timetuple())
            Mi_data_proceesed['square_id'].append(square_id)
            Mi_data_proceesed['timestamp'].append(end_timestamp)
            Mi_data_proceesed['sms_in_activity'].append(
                sms_in_activity / (time_interval / 10))
            Mi_data_proceesed['sms_out_activity'].append(
                sms_out_activity / (time_interval / 10))
            Mi_data_proceesed['call_in_activity'].append(
                call_in_activity / (time_interval / 10))
            Mi_data_proceesed['call_out_activity'].append(
                call_out_activity / (time_interval / 10))
            Mi_data_proceesed['internat_traffic_activity'].append(
                temp_activity / (time_interval / 10))
            # print(Mi_data_proceesed['square_id'][-1],end_time_str,Mi_data_proceesed['internat_traffic_activity'][-1],Mi_data_proceesed['sms_out_activity'][-1])

            # update end_time
            end_time = end_time + timedelta(minutes=time_interval)
            temp_activity = 0
    return Mi_data_proceesed


def clean_data(Mi_data_proceesed):
    previous_internat_traffic_activity = 0
    len_of_Mi_data_proceesed_internet_traffic = len(Mi_data_proceesed['internat_traffic_activity'])
    for i, element in enumerate(Mi_data_proceesed['internat_traffic_activity']):
        if Mi_data_proceesed['internat_traffic_activity'][i] < previous_internat_traffic_activity * 1 / 100 or int(Mi_data_proceesed['internat_traffic_activity'][i]) == 0:
            try:
                next_value = Mi_data_proceesed[
                    'internat_traffic_activity'][i + 1]
                average = (previous_internat_traffic_activity + next_value) / 2
            except:
                average = previous_internat_traffic_activity
            # print(len_of_Mi_data_proceesed, i, i + 1)
            if len_of_Mi_data_proceesed_internet_traffic > i + 1:
                next_squire_id = Mi_data_proceesed['square_id'][i + 1]
                if Mi_data_proceesed['square_id'][i] == next_squire_id:

                    print('find dirty data!! id:{} timestamp:{} before:{} next:{} origin:{} new value:{}'.format(
                        Mi_data_proceesed['square_id'][i],
                        Mi_data_proceesed['timestamp'][i],
                        previous_internat_traffic_activity,
                        next_value,
                        Mi_data_proceesed['internat_traffic_activity'][i],
                        average))
                    Mi_data_proceesed['internat_traffic_activity'][i] = average

        previous_internat_traffic_activity = Mi_data_proceesed['internat_traffic_activity'][i]

    return Mi_data_proceesed


def process_data_to_mildan_grid(Mi_data_proceesed):
    grid_size = 10001
    grid_row_num = 100
    grid_column_num = 100
    features_num = 7
    grid_list = [None] * grid_size

    for i in range(len(grid_list)):
        grid_list[i] = []

    for i, _id in enumerate(Mi_data_proceesed['square_id']):
        timestamp = Mi_data_proceesed['timestamp'][i]
        date_time = set_time_zone(timestamp)

        square_id = int(Mi_data_proceesed['square_id'][i])
        sms_in_activity = float(Mi_data_proceesed['sms_in_activity'][i])
        sms_out_activity = float(Mi_data_proceesed['sms_out_activity'][i])
        call_in_activity = float(Mi_data_proceesed['call_in_activity'][i])
        call_out_activity = float(Mi_data_proceesed['call_out_activity'][i])
        internat_traffic_activity = float(
            Mi_data_proceesed['internat_traffic_activity'][i])

        feature_element = [_id, timestamp, sms_in_activity, sms_out_activity,
                           call_in_activity, call_out_activity, internat_traffic_activity]
        # print(i,square_id,date_time_covert_to_str(date_time),feature_element)
        grid_list[_id].append(feature_element)

    # if 10 minutes in a record ,should be 144 a day
    each_grid_length = len(grid_list[9999])
    print('each_grid_length', each_grid_length)
    array_size = [each_grid_length, grid_row_num,
                  grid_column_num, features_num]
    X = np.zeros(array_size)
    for square_id in range(1, grid_size + 1):
        row = 99 - int(square_id / grid_row_num)  # row mapping in milan grid
        column = square_id % grid_column_num - 1  # column mapping in milan grid

        for bach_index in range(each_grid_length):
            try:
                #print('grid list',square_id,bach_index,grid_list[square_id][bach_index])
                X[bach_index][row][column] = grid_list[square_id][bach_index]
                # print('X',square_id,bach_index,X[bach_index][row][column])

            except:
                X[bach_index][row][column] = np.zeros([features_num])
    print(X.shape)
    return X


def save_array(x_array, out_file):
    print('saving file to {}...'.format(out_file))
    np.save(out_file, x_array, allow_pickle=True)


def load_array(input_file):
    print('loading file from {}...'.format(input_file))
    X = np.load(input_file + '.npy')
    return X


def set_time_zone(timestamp):
    date_time = datetime.utcfromtimestamp(float(timestamp))
    date_time = date_time.replace(tzinfo=UTC_timezone)
    date_time = date_time.astimezone(Mi_timezone)
    return date_time


def date_time_covert_to_str(date_time):
    return date_time.strftime('%Y-%m-%d %H:%M:%S')


def load_data_from_file(file_path):

    Mi_data = {
        'square_id': [],
        'timestamp': [],
        'sms_in_activity': [],
        'sms_out_activity': [],
        'call_in_activity': [],
        'call_out_activity': [],
        'internat_traffic_activity': []
    }

    Mi_data_proceesed = {
        'square_id': [],
        'timestamp': [],
        'sms_in_activity': [],
        'sms_out_activity': [],
        'call_in_activity': [],
        'call_out_activity': [],
        'internat_traffic_activity': []
    }
    # maybe use panda
    with open(file_path, 'r') as f:
        print('start to load data from {}..'.format(file_path))
        for line in f.readlines():
            split_line = line.split('\t')

            square_id = int(split_line[0].strip())
            timestamp = int(split_line[1].strip()) / 1000
            country_code = int(split_line[2].strip())
            sms_in_activity = float(split_line[3].strip()) if split_line[
                3].strip() else 0
            sms_out_activity = float(split_line[4].strip()) if split_line[
                4].strip() else 0
            call_in_activity = float(split_line[5].strip()) if split_line[
                5].strip() else 0
            call_out_activity = float(split_line[6].strip()) if split_line[
                6].strip() else 0
            internat_traffic_activity = float(split_line[7].strip()) if split_line[
                7].strip() else 0

            date_time = datetime.utcfromtimestamp(float(timestamp))
            date_time = date_time.replace(tzinfo=UTC_timezone)
            date_time = date_time.astimezone(Mi_timezone)
            date_time = date_time_covert_to_str(date_time)

            if internat_traffic_activity != 0:
                Mi_data['square_id'].append(square_id)
                Mi_data['timestamp'].append(timestamp)
                Mi_data['sms_in_activity'].append(sms_in_activity)
                Mi_data['sms_out_activity'].append(sms_out_activity)
                Mi_data['call_in_activity'].append(call_in_activity)
                Mi_data['call_out_activity'].append(call_out_activity)
                Mi_data['internat_traffic_activity'].append(
                    internat_traffic_activity)

                # print(date_time,timestamp,square_id,internat_traffic_activity)

    Mi_data_proceesed = combine_data(Mi_data, Mi_data_proceesed)
    Mi_data_proceesed = clean_data(Mi_data_proceesed)
    del Mi_data
    X_image = process_data_to_mildan_grid(Mi_data_proceesed)

    output_dir = os.path.dirname(file_path) + '/data_preproccessing_10/'
    output_filename = 'output_' + \
        os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_array(X_image, output_dir + output_filename)
    #X_image = load_array(ouput_dir+ouput_file)


for input_dir in input_dir_list:
    filelist = list_all_input_file(input_dir)
    filelist.sort()

    print("filelist length:{}".format(len(filelist)))
    for file_name in filelist:
        load_data_from_file(input_dir + file_name)
