import pandas as pd
import numpy
import os
import csv

elements = 1000

sensors = ['T','RA','LA','RL','LL']
parameters = ['acc','gyro','mag']
final_sensors = []
for sensor in sensors:
    for parameter in parameters:
        name = sensor
        name = name + '_x' + parameter
        final_sensors.append(name)
        name = sensor
        name = name + '_y' + parameter
        final_sensors.append(name)
        name = sensor
        name = name + '_z' + parameter
        final_sensors.append(name)

def load_final_mrl(feature_no):
    file_name = 'final_mrl_' + str(feature_no) + '.csv'
    locat = os.getcwd()
    locat = locat + '/mrls/' + file_name
    dataframe = pd.read_csv(locat,header=None)
    X = dataframe.iloc[0,:].to_numpy()
    return X

def check_no_of_elements(dataset_no,elements):
    file_name = 'dataset' + str(dataset_no) + '.csv'
    locat = os.getcwd()
    locat = locat + '/' + file_name
    dataframe = pd.read_csv(locat)
    arr = []
    for i in range(19):
        arr.append(0)
    for i in range(elements):
        z = int(dataframe.iat[i,124])
        arr[z-1] = arr[z-1] + 1
    for i in range(19):
        print("dataset no",dataset_no," label no",i+1," count is ",arr[i])