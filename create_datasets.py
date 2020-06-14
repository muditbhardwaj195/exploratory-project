import pandas as pd
import numpy as np
import os
import csv


Training_percent = 0.7

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

# for feature_no in range(45):
#     csv_file_name = "dataset_" + final_sensors[feature_no - 1] + ".csv"
#     print(csv_file_name)

final_list = []

def process_call(location,label_id):
    directory = location
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = directory + '/' + filename
            if(label_id in [1,2,3,4,16]):
                process_feature_file(path,label_id)
            # process_feature_file(path,label_id)
        else: 
            path = directory + '/' + filename
            isDirectory = os.path.isdir(path)
            if isDirectory:
                if filename.startswith("a"):
                    s = filename[-2:]
                    label_id_temp = int(s)
                    process_call(path,label_id_temp)
                else:
                    process_call(path,label_id)

def process_feature_file(file_path,label_id):
    print("Reading file of label id... ",label_id)
    dataframe = pd.read_csv(file_path,header=None)
    no_of_rows,no_of_columns = dataframe.shape
    arr = list()
    for col in range(no_of_columns):
        for row in range(no_of_rows):
            arr.append(dataframe.iat[row,col])
    arr.append(label_id)
    final_list.append(arr)
    

def write_to_csv_file():
    for x in range(1):
        print("writing Complete database")
        csv_file_name = "complete_dataset.csv"
        with open(csv_file_name, 'w', newline='') as file:
            csv_writer = csv.writer(file)
        
            for i in range(len(final_list)):
                row = list()
                row = final_list[i]
                csv_writer.writerow(row)

def jumble_datasets_elements():
    print("Jumbling Complete database")
    file_name = 'complete_dataset.csv'
    locat = os.getcwd()
    locat = locat + '/' + file_name
    dataframe = pd.read_csv(locat,header=None)
    dataframe = dataframe.sample(frac = 1)
    tmp_dataframe = dataframe.iloc[:,:]
    x = tmp_dataframe.to_numpy()

    csv_file_name = "complete_dataset.csv"
    with open(csv_file_name, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        for i in range(len(x)):
            row = list()
            row = x[i]
            csv_writer.writerow(row)

def create_sensor_dataset():
    print("Loading complete dataset...")
    file_name = 'complete_dataset.csv'
    locat = os.getcwd()
    locat = locat + '/' + file_name
    dataframe = pd.read_csv(locat,header=None)
    rows,cols = dataframe.shape
    rows = int(rows*Training_percent)
    for sensor in range(45):
        print("Creating dataset for sensor ",sensor+1)
        starting_ind = int(125*sensor)
        ending_ind = int(starting_ind + 125)
        sensor_final = []
        for i in range(rows):
            j = starting_ind
            ttt = list()
            while j < ending_ind:
                ttt.append(dataframe.iat[i,j])
                j += 1
            ttt.append(dataframe.iat[i,-1])
            sensor_final.append(ttt)
        csv_file_name = "dataset_" + str(sensor+1) + ".csv"
        with open(csv_file_name, 'w', newline='') as file:
            csv_writer = csv.writer(file)

            for i in range(len(sensor_final)):
                row = list()
                row = sensor_final[i]
                csv_writer.writerow(row)

def create_test_cases():
    print("Loading complete dataset...")
    file_name = 'complete_dataset.csv'
    locat = os.getcwd()
    locat = locat + '/' + file_name
    dataframe = pd.read_csv(locat,header=None)
    temp = []
    rows,cols = dataframe.shape
    rows_start = int(rows*Training_percent)
    for i in range(rows):
        if i > rows_start:
            tt = list()
            for j in range(cols):
                for sensor in [1,10,19,28,37]:
                    starting_ind = int(sensor-1)*125
                    ending_ind = int(starting_ind + 125)
                    if (j >= starting_ind) and ( j < ending_ind):
                        tt.append(dataframe.iat[i,j])
                
                if j == int(cols-1):
                    tt.append(dataframe.iat[i,j])
            temp.append(tt)
    
    #now write to test case file
    print("writing Test case file database")
    csv_file_name = "merged_dataset.csv"
    with open(csv_file_name, 'w', newline='') as file:
        csv_writer = csv.writer(file)
    
        for i in range(len(temp)):
            row = list()
            row = temp[i]
            csv_writer.writerow(row)



def make_test_cases():
    print("Starting to make test case file: merged_dataset.csv")
    locat = os.getcwd()
    process_call(locat,-1)
    write_to_csv_file()
    jumble_datasets_elements()
    create_sensor_dataset()
    create_test_cases()
    print("test case file created")

# make_test_cases()


