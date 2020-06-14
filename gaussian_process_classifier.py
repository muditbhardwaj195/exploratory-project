import pandas as pd
import numpy
import os
import csv
from load_datasets import load_datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle

STARTING_FRACTION = 0.4
ALPHA = 0.8
alpha = 0.8
Training_percent = 0.7
ANSWER = list()

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


mrl_file_heading = []
for i in range(19):
        s = 'A' + str(i+1)
        mrl_file_heading.append(s)

def check_probabilities_for_f(orignal_probability,new_probability,alpha,labels,target_label):
        print("Checking for acceptable probabilities")
        for i, orignal_prob in enumerate(orignal_probability):
                orignal_proba_array = list(orignal_prob)
                new_proba_array = list(new_probability[i])

                if labels[i] == target_label:
                        if((alpha*max(orignal_proba_array)) > max(new_proba_array)):
                                return 0
        
        return 1


def calculate_t(dataset_no):
        print("Starting to find orignal_labels ans for dataset no", dataset_no)
        X,y = load_datasets(dataset_no)
        rows,col = X.shape
        kernel = 1.0 * RBF(1.0)
        ROW = int(Training_percent*rows)
        ROW = 800
        print("Starting $")
        gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X[:ROW,:], y[:ROW])
        print("Successfully trained ",dataset_no)
        print("Starting predicting data for full length")
        orignal_probability = gpc.predict_proba(X[:ROW,:])
        print("Orignal_probability array calculated for dataset_no",dataset_no)

        mrl = [None for _ in range(5)]

        current_pos = int(STARTING_FRACTION*col)

        while 1:
                gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X[:ROW,:current_pos], y[:ROW])
                new_probability = gpc.predict_proba(X[:ROW,:current_pos])

                print("Probabilities calculated for current value of f = ",current_pos)

                for i in range(5):
                        value_mrl = mrl[i]
                        if not(value_mrl):
                                if i == 4:
                                        temporary = 16
                                else:
                                        temporary = i+1
                                if(check_probabilities_for_f(orignal_probability,new_probability,alpha,y,temporary)):
                                        mrl[i] = current_pos
                                        print("F for label" ,temporary, " is ", current_pos)
                                        # print("Saving model")
                                        # s = 'label_id' + str(i+1) + "component" + str(dataset_no)
                                        # filename = 'models/' + s + '.sav'
                                        # pickle.dump(gpc, open(filename, 'wb'))
                
                all_completed = 1
                for value_mrl in mrl:
                        if not(value_mrl):
                                all_completed = 0
                if all_completed:
                        break
                
                current_pos = current_pos + 5
        return mrl

def calculate_MRD():
        for no in [1,10,19,28,37]:
                print("Calculating MRD for sensor ",no)
                z = calculate_t(no)
                print(z)
                csv_file_name = "mrls/final_mrl_" + str(no) + ".csv"
                with open(csv_file_name, 'w',newline = '') as file:
                        csv_writer = csv.writer(file)
                        # csv_writer.writerow(mrl_file_heading)
                        csv_writer.writerow(z)
                # ANSWER.append(z)
                print("Finished calculating MRD for sensor ",no)

def save_in_file():
        csv_file_name = "mrls/final_mrl.csv"
        with open(csv_file_name, 'w', newline='') as file:
            csv_writer = csv.writer(file)
        
            for i in range(len(ANSWER)):
                row = list()
                row = ANSWER[i]
                csv_writer.writerow(row)

