import pandas as pd
import numpy
import os
import math
import csv
from load_datasets import load_datasets
from load_final_mrl import load_final_mrl
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle

sampling_rate = {
    1 : 1.0000,
    10 : 0.9000,
    19 : 0.8000,
    28 : 0.65000,
    37 : 0.55000, 
}
total_sensors = 5
tree_nodes = []
for i in range(110):
    arr = list()
    tree_nodes.append(arr)
level_starting_index = list()
tree_depth = 1
ROWS = 800
kernel = 1.0 * RBF(1.0)
classified = list()

def create_child_nodes(num,depth):
    parent_node_size = int(len(tree_nodes[num]))
    global tree_depth
    if tree_depth < depth:
        tree_depth = depth
    if(parent_node_size == 1):
        return 0
    child_left = int(2*num + 1)
    child_right = int(2*num + 2)
    parent_arry = tree_nodes[num]
    diff_array = list()
    for i in range(parent_node_size - 1):
        a = sampling_rate[parent_arry[i]]
        a = round(a,2)
        b = sampling_rate[parent_arry[i+1]]
        b = round(b,2)
        c = a - b
        c = round(c,2)
        diff_array.append(c)
    maxa = max(diff_array)
    count = 0
    ind = 0
    for i in range(int(len(diff_array))):
        if(diff_array[i] == maxa):
            count += 1
            ind = i
    if(count == int(len(diff_array))):
        #split equally
        if(int(parent_node_size%2) == 1):
            left_arr = list()
            mid = int(parent_node_size/2)
            left_arr = tree_nodes[num][:mid+1]
            right_arr = tree_nodes[num][-mid:]
            tree_nodes[child_left] = left_arr
            tree_nodes[child_right] = right_arr
            create_child_nodes(child_left,depth + 1)
            create_child_nodes(child_right,depth + 1)
        else:
            left_arr = list()
            mid = int(parent_node_size/2)
            left_arr = tree_nodes[num][:mid]
            right_arr = tree_nodes[num][-mid:]
            tree_nodes[child_left] = left_arr
            tree_nodes[child_right] = right_arr
            create_child_nodes(child_left,depth + 1)
            create_child_nodes(child_right,depth + 1)
    else:
        left_arr = list()
        left_arr = tree_nodes[num][:ind+1]
        remain = int(parent_node_size) - int(ind + 1)
        right_arr = list()
        right_arr = tree_nodes[num][-remain:]
        tree_nodes[child_left] = left_arr
        tree_nodes[child_right] = right_arr
        create_child_nodes(child_left,depth + 1)
        create_child_nodes(child_right,depth + 1)

#making a binary tree of sensors with different sampling rate

def make_binary_tree():
    print("Creating binary tree")
    tree_nodes[0] = [1,10,19,28,37]
    create_child_nodes(0,0)


#assigning starting index of each level
def assign_starting_index_of_level():
    sum = 0
    global tree_depth
    for level in range(int(tree_depth + 1)):
        level_starting_index.append(int(sum))
        sum = sum + 2**level

def tell_points_required(feature,label):
    X = load_final_mrl(feature)
    if(label == 16):
        return X[4]
    else:
        return X[label-1]

def tell_min_mrl_value(feature):
    X = load_final_mrl(feature)
    mini = min(X)
    return mini

def apply_gaussian_classifier(feature,col_required,array_to_predict):
    Main_X, Main_Y = load_datasets(feature)
    print("Starting Gausian")
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(Main_X[:ROWS,:col_required], Main_Y[:ROWS])
    print("Successfully Trained :)")
    orignal_probability = gpc.predict_proba(array_to_predict)
    maxa = max(orignal_probability[0])
    for j in range(5):
        if(orignal_probability[0][j] == maxa):
            index = j
    if(index == 0):
        return 1
    elif(index == 1):
        return 2
    elif(index == 2):
        return 3
    elif(index == 3):
        return 4
    else:
        return 16


def classify_time_series(test_time_series):
    level = tree_depth
    final_list = []
    print("Starting time series classification")
    while level > 0:
        starting_ind = level_starting_index[level]

        total_elements_at_present_level = 2**level
        total_pairs_at_present_level = int(total_elements_at_present_level/2)

        for i in range(total_pairs_at_present_level):
            left_ans = 0
            right_ans = 0
            points_taken_left = 0
            points_taken_right = 0

            #left side index will be starting_ind + (2*i)
            left_arr = list()
            left_index = starting_ind + (2*i)
            left_arr = tree_nodes[left_index]
            left_arr_size = int(len(left_arr))
            if(left_arr_size == 0):
                continue
            print("Presently at level : ",level," of the tree with left array and right array at index ",left_index," ",left_index+1)
            print("Left array is :",left_arr)
            if(left_arr_size == 1):
                only_element = int(left_arr[0])
                min_mrl = tell_min_mrl_value(only_element)
                arrz = []
                arrz.append(test_time_series[only_element][:min_mrl])
                left_ans = apply_gaussian_classifier(only_element,min_mrl,arrz)
                points_taken_left = min_mrl

            else:
                first_element = int(left_arr[0])
                min_mrl = tell_min_mrl_value(first_element)
                arrz = []
                arrz.append(test_time_series[first_element][:min_mrl])
                left_ans = apply_gaussian_classifier(first_element,min_mrl,arrz)
                points_taken_left = min_mrl

                for j in range(left_arr_size):
                    if(j):
                        feature = int(left_arr[j])
                        points_available_ratio = sampling_rate[feature]/sampling_rate[left_arr[j-1]]
                        points_available = int(points_available_ratio*points_taken_left)
                        points_required = tell_points_required(feature,left_ans)
                        if(points_required <= points_available):
                            arrz = []
                            arrz.append(test_time_series[feature][:points_available])
                            left_ans = apply_gaussian_classifier(feature,points_available,arrz)
                            points_taken_left = points_available
                        else:
                            points_taken_left = points_required
                            arrz = []
                            arrz.append(test_time_series[feature][:points_required])
                            left_ans = apply_gaussian_classifier(feature,points_required,arrz)


            #right side index will be starting_ind + (2*i) + 1
            right_arr = list()
            right_index = starting_ind + (2*i) + 1
            right_arr = tree_nodes[right_index]
            right_arr_size = int(len(right_arr))
            print("Right array is : ",right_arr)
            if(right_arr_size == 1):
                only_element = int(right_arr[0])
                min_mrl = tell_min_mrl_value(only_element)
                arrz = []
                arrz.append(test_time_series[only_element][:min_mrl])
                right_ans = apply_gaussian_classifier(only_element,min_mrl,arrz)
                points_taken_right = min_mrl

            else:
                first_element = int(right_arr[0])
                min_mrl = tell_min_mrl_value(first_element)
                arrz = []
                arrz.append(test_time_series[first_element][:min_mrl])
                right_ans = apply_gaussian_classifier(first_element,min_mrl,arrz)
                points_taken_right = min_mrl

                for j in range(right_arr_size):
                    if(j):
                        feature = int(right_arr[j])
                        points_available_ratio = sampling_rate[feature]/sampling_rate[right_arr[j-1]]
                        points_available = int(points_available_ratio*points_taken_right)
                        points_required = tell_points_required(feature,right_ans)
                        if(points_required <= points_available):
                            arrz = []
                            arrz.append(test_time_series[feature][:points_available])
                            right_ans = apply_gaussian_classifier(feature,points_available,arrz)
                            points_taken_right = points_available
                        else:
                            points_taken_right = points_required
                            arrz = []
                            arrz.append(test_time_series[feature][:points_required])
                            right_ans = apply_gaussian_classifier(feature,points_required,arrz)

            print("Left_ans :",left_ans," Right_ans : ",right_ans)
            if(left_ans == right_ans):
                print("(: Left_ans is same as Right_ans :)")
                print("Finally time series classified to label ", right_ans," at this node.")
                temp = list()
                temp.append(int(right_ans))
                temp.append(max(points_taken_left,points_taken_right))
                final_list.append(temp)

            else:
                print("Left_ans is different from Right_ans :( ")
                print("Starting again on right array")
                points_taken = 0
                for j in range(right_arr_size):
                    if(j == 0):
                        feature = int(right_arr[j])
                        points_required = tell_points_required(feature,left_ans)
                        arrz = []
                        arrz.append(test_time_series[feature][:points_required])
                        right_ans = apply_gaussian_classifier(feature,points_required,arrz)
                        points_taken = points_required
                    else:
                        feature = int(right_arr[j])
                        points_available_ratio = sampling_rate[feature]/sampling_rate[right_arr[j-1]]
                        points_available = int(points_available_ratio*points_taken)
                        points_required = tell_points_required(feature,right_ans)
                        if(points_required <= points_available):
                            arrz = []
                            arrz.append(test_time_series[feature][:points_available])
                            right_ans = apply_gaussian_classifier(feature,points_available,arrz)
                            points_taken = points_available
                        else:
                            points_taken = points_required
                            arrz = []
                            arrz.append(test_time_series[feature][:points_required])
                            right_ans = apply_gaussian_classifier(feature,points_required,arrz)
                
                temp = list()
                temp.append(int(right_ans))
                temp.append(points_taken)
                final_list.append(temp)
                print("Finally time series classified to label ", right_ans," at this node.")

        level = level - 1


    if(level == 0):
        print("Reached Level 0 :)")
        mid_arr = list()
        mid_index = 0
        mid_ans = 0
        points_taken_mid = 0
        mid_arr = tree_nodes[mid_index]
        mid_arr_size = int(len(mid_arr))
        print("At level 0, array is : ",mid_arr)
        if(mid_arr_size == 1):
            only_element = int(mid_arr[0])
            min_mrl = tell_min_mrl_value(only_element)
            arrz = []
            arrz.append(test_time_series[only_element][:min_mrl])
            mid_ans = apply_gaussian_classifier(only_element,min_mrl,arrz)
            points_taken_mid = min_mrl

        else:
            first_element = int(mid_arr[0])
            min_mrl = tell_min_mrl_value(first_element)
            arrz = []
            arrz.append(test_time_series[first_element][:min_mrl])
            right_ans = apply_gaussian_classifier(first_element,min_mrl,arrz)
            points_taken_mid = min_mrl

            for j in range(mid_arr_size):
                if(j):
                    feature = int(mid_arr[j])
                    points_available_ratio = sampling_rate[feature]/sampling_rate[mid_arr[j-1]]
                    points_available = int(points_available_ratio*points_taken_mid)
                    points_required = tell_points_required(feature,mid_ans)
                    if(points_required <= points_available):
                        arrz = []
                        arrz.append(test_time_series[feature][:points_available])
                        mid_ans = apply_gaussian_classifier(feature,points_available,arrz)
                        points_taken_mid = points_available
                    else:
                        points_taken_mid = points_required
                        arrz = []
                        arrz.append(test_time_series[feature][:points_required])
                        mid_ans = apply_gaussian_classifier(feature,points_required,arrz)
        print("Finally time series classified to label ", mid_ans," at this node.")
        temp = list()
        temp.append(int(mid_ans))
        temp.append(points_taken_mid)
        final_list.append(temp)

        #calculating maximum times appearing 
        count = list()
        indexs = list()
        for i in range(5):
            count.append(0)
            indexs.append(0)
        maxa = 0
        ans_ind = -1
        for k in range(len(final_list)):
            ar = final_list[k]
            num = ar[0]
            tmp_ind = ar[1]
            if(num == 16):
                count[4] += 1
                indexs[4] = max(int(indexs[4]),int(tmp_ind))
            else:
                count[num-1] += 1
                indexs[num-1] = max(int(indexs[num-1]),int(tmp_ind))
        
        for k in range(5):
            if maxa < count[k]:
                maxa = count[k]
                ans_ind = k
        
        ans_points_req = 0
        ans_label = 0
        if ans_ind == 4:
            ans_label =  16
        else:
            ans_label =  ans_ind+1

        for k in range(len(final_list)):
            ar = final_list[k]
            num = ar[0]
            tmp_ind = ar[1]
            if(int(num) == int(ans_label)):
                ans_points_req = max(int(ans_points_req),int(tmp_ind))

        return ans_label,ans_points_req
        

def main_function(test_time_series):
    make_binary_tree()
    assign_starting_index_of_level()
    z,y = classify_time_series(test_time_series)
    return z,y

#driving function of time_series_prediction.py
def check():
    file_name = 'merged_dataset.csv'
    locat = os.getcwd()
    locat = locat + '/' + file_name
    dataframe = pd.read_csv(locat,header=None)
    print("Test file loaded successfully ")
    for j in range(200):
        if j < 0:
            continue
        print("Time Series id: ",j)
        X = dataframe.iloc[j,:-1].to_numpy()
        Y = dataframe.iloc[j,-1]
        print(X.shape,Y)
        tmp_arr = []
        for i in range(46):
            zz = []
            if(i == 1):
                zz = X[:125]
            elif(i == 10):
                zz = X[125:250]
            elif(i == 19):
                zz = X[250:375]
            elif(i == 28):
                zz = X[375:500]
            elif(i == 37):
                zz = X[500:625]
            tmp_arr.append(zz)

        z,y = main_function(tmp_arr)
        at = list()
        at.append(dataframe.iat[j,-1])
        at.append(z)
        at.append(y)
        classified.append(at)
        print("Predicted class label for this time series is : ",z," and min points required is : ",y)
        print("Writing this time series result to predicted_class.csv")
        csv_file_name = "predicted_class.csv"
        with open(csv_file_name, 'a', newline='') as file:
            csv_writer = csv.writer(file)

            for i in range(1):
                row = list()
                row = at
                csv_writer.writerow(row)

    print("Time series classes they are classified into are:")
    for j in range(100):
        print("Orignal label: ",dataframe.iat[j,-1]," Predicted label:",classified[j][0]," Points required: ",classified[j][1])
    
    print("Finished")

def calculate_accuracy():
    print("Calculating Accuracy...")
    file_name = 'predicted_class.csv'
    locat = os.getcwd()
    locat = locat + '/' + file_name
    dataframe = pd.read_csv(locat)
    truly_predicted = 0
    wrongly_predicted = 0
    total_test_cases = 0
    row,col = dataframe.shape
    z = 0
    count = 0
    for i in range(row):
        x = int(dataframe.iat[i,0])
        y = int(dataframe.iat[i,1])
        z += int(dataframe.iat[i,2])
        count +=int(125)
        if(x == y):
            truly_predicted += 1
        else:
            wrongly_predicted += 1
        total_test_cases += 1
    accuracy = (truly_predicted*100)/total_test_cases
    earliness = ((count - z)*100)/count
    print("Accuracy is : ", accuracy)
    print("Earliness is : ",earliness)
    print("finished")

