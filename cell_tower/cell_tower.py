import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime
import json
from descartes import PolygonPatch
import random


def filter_cell_tower():
    source_file = '/home/mldp/Download/cell.csv'
    target_file = './cell_tower.csv'
    total_record = 0
    with open(source_file, 'r') as csvfile:
        csv_target = open(target_file, 'w')
        cellreader = csv.reader(csvfile)
        writer = csv.writer(csv_target, delimiter=',')
        for row_index, row in enumerate(cellreader):

            if row_index == 0:
                col_name_list = row
                print(col_name_list)
                continue
            radio = row[0]
            mcc = int(row[1])
            mnc = row[2]
            lon = float(row[6])
            lat = float(row[7])
            timestamp = int(row[11])
            if mcc != 222 or mnc != '1':
                continue

            if 9 <= lon <= 9.3 and 45.36 <= lat <= 45.56:
                limit_date_obj = datetime(2014, 6, 1)
                create_date_obj = datetime.fromtimestamp(timestamp)
            if limit_date_obj > create_date_obj:
                    str_time = create_date_obj.strftime('%Y-%m-%d')
                    print(mcc, mnc, 'radio:{} lon:{} lat:{} create_time:{}'.format(
                        radio, lon, lat, str_time))
                    writer.writerow([radio, lon, lat])
                    total_record += 1

        csv_target.close()
    print('total record:', total_record)


def plot_filtered_cell_tower():
    source_file = './cell_tower.csv'
    df = pd.read_csv(source_file, names=['radio', 'Lon', 'Lat'])
    ex = df.plot(kind='scatter', x='Lon', y='Lat', color='w', edgecolors='darkblue', title='Macro cell')
    ex.set_xlabel('Longitude')
    ex.set_ylabel('Latitude')
    ex.grid()
    plt.show()


def assign_grid_to_cell_tower():
    cell_tower_source_file = './cell_tower.csv'
    grid_geo_source_file = './milano-grid.geojson'
    target_file = './cell_tower_with_grid.txt'

    def get_geo_data():
        grid_list = []
        with open(grid_geo_source_file, 'r') as geofile:
            geo_data = json.load(geofile)

        for grid in geo_data['features']:
            grid_id = grid['properties']['cellId']
            grid_coord = grid['geometry']['coordinates'][0]
            x = (grid_coord[0][0] + grid_coord[2][0]) / 2
            y = (grid_coord[0][1] + grid_coord[2][1]) / 2
            grid_list.append({'cellid': grid_id, 'center_coord': [x, y]})

        return grid_list

    def get_cell_tower():
        tower_list = []
        with open(cell_tower_source_file, 'r') as f:
            cellreader = csv.reader(f)
            for row_index, row in enumerate(cellreader):
                radio = row[0]
                Lon = float(row[1])
                Lat = float(row[2])
                tower_list.append(
                    {'index': row_index, 'radio': radio, 'coord': [Lon, Lat], 'grid': []})
        return tower_list

    def calculate_distance(grid_coord, tower_list):
        min_distance = math.inf
        min_distance_tower_index = 0
        for tower in tower_list:
            tower_coord = tower['coord']
            # print(tower_coord)
            # print(grid_coord)
            distance = math.sqrt(
                (tower_coord[0] - grid_coord[0]) ** 2 + (tower_coord[1] - grid_coord[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                min_distance_tower_index = tower['index']

        return min_distance_tower_index

    grid_list = get_geo_data()
    tower_list = get_cell_tower()

    for grid in grid_list:
        tower_index = calculate_distance(grid['center_coord'], tower_list)
        tower_list[tower_index]['grid'].append(grid['cellid'])
        print(tower_index)

    with open(target_file, 'w') as outfile:
        json.dump(tower_list, outfile, sort_keys=False, indent=4)


def plot_cell_tower_and_grid():
    cell_tower_with_grid = './cell_tower_with_grid.txt'
    grid_geo_source_file = './milano-grid.geojson'

    def get_geo_data():
        grid_list = []
        with open(grid_geo_source_file, 'r') as geofile:
            geo_data = json.load(geofile)

        for grid in geo_data['features']:
            grid_id = grid['properties']['cellId']
            grid_coord = grid['geometry']
            # grid_coord = [(float(x), float(y))for x, y in grid_coord]
            # x = (grid_coord[0][0] + grid_coord[2][0]) / 2
            # y = (grid_coord[0][1] + grid_coord[2][1]) / 2
            grid_list.append({'cellid': grid_id, 'coordinate': grid_coord})

        return grid_list

    def get_cell_tower_with_grid():
        with open(cell_tower_with_grid, 'r') as f:
            cell_grid = json.load(f)
        return cell_grid

    def search_grid_id_coord(grid_id, grid_list):
        for each_grid in grid_list:
            if grid_id != each_grid['cellid']:
                continue
            # print(each_grid['coordinate'])
            return each_grid['coordinate']

    grid_list = get_geo_data()
    cell_grid = get_cell_tower_with_grid()
    cell_grid.sort(key=lambda x: x['coord'])
    fig, ax = plt.subplots()
    colors = ['aqua', 'azure', 'beige', 'black', 'blue', 'brown', 'chartreuse', 'chocolate', 'coral', 'crimson', 'cyan', 'darkblue', 'darkgreen', 'fuchsia', 'gold', 'goldenrod', 'green', 'indigo', 'grey', 'khaki', 'lavender', 'lightblue',
              'lightgreen', 'lime', 'magenta', 'maroon', 'navy', 'olive', 'orange', 'orangered', 'orchid', 'pink', 'plum', 'purple', 'red', 'salmon', 'sienna', 'silver', 'tan', 'teal', 'tomato', 'turquoise', 'violet', 'wheat', 'yellow', 'yellowgreen']
    for index, cell in enumerate(cell_grid):
        # print(cell['coord'][0], cell['coord'][1])
        if len(cell['grid']) == 0:
            continue
        if cell['radio'] == 'GSM':
            ax.scatter(x=cell['coord'][0], y=cell[
                       'coord'][1], marker='.', color='r')
        elif cell['radio'] == 'WCDM':
            ax.scatter(x=cell['coord'][0], y=cell[
                       'coord'][1], marker='.', color='r')
        else:
            ax.scatter(x=cell['coord'][0], y=cell[  # UMTS
                       'coord'][1], marker='.', color='r')
            # print(cell['radio'])

        fc_color = colors[index % len(colors)]
        # bc_color = random.choice(colors)
        for grid_id in cell['grid']:
            grid_id_coord = search_grid_id_coord(grid_id, grid_list)
            ax.add_patch(PolygonPatch(grid_id_coord, fc=fc_color,
                                      ec=fc_color, alpha=0.3, zorder=2))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Macro cell coverage')
    plt.show()


# filter_cell_tower()
# plot_filtered_cell_tower()
# assign_grid_to_cell_tower()

plot_cell_tower_and_grid()
