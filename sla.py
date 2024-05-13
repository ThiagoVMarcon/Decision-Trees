import numpy as np
import pandas as pd
 
 #discretizes the numerical attributes of the DataFrame
def discretize(data):
        num_of_columns = data.shape[1]
        for column in range(num_of_columns):
            try:
                data.iloc[0,column] = float(data.iloc[0,column])
                do_Discretize(column, data)
            except ValueError:
                pass
    
    # discretizes a single numerical column of the DataFrame (helper of discretize)
def do_Discretize(column, data):
        column_name = data.columns[column]
        std_dev = np.std(data[column_name])
        num_points = len(data[column_name])
        bin_width = 3.5 * std_dev / (num_points ** (1/3))
        data_range = data[column_name].max() - data[column_name].min()
        num_bins = int(np.ceil(data_range / bin_width))
        labels = [f'[{data[column_name].min() + i * bin_width:.2f}-{data[column_name].min() + (i + 1) * bin_width:.2f})' for i in range(num_bins)]
        data[column_name] = pd.qcut(data[column_name], q=num_bins, labels=labels)



def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    total_entr = 0
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #entropy of the class
        total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset
    
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    #print(feature_value_data)
    class_count = feature_value_data.shape[0]
    entropy = 0
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value

    return round(calc_total_entropy(train_data, label, class_list) - feature_info, 6) #calculating information gain by subtracting

def motherfucker(data, n_colums):
    lista = []
    unique_goal_name = data.columns[ncols-2]
    unique_goal = data[unique_goal_name].unique()
    for i in range(0,data.shape[1] - 1):
        column_name = data.columns[i]
        #print(column_name)
        x = calc_info_gain(column_name, data, unique_goal_name, unique_goal)
        lista.append(x)
    return lista

with open("Datasets/weather.csv") as x:
    ncols = len(x.readline().split(','))
data = pd.read_csv("Datasets/weather.csv", usecols=range(1,ncols))
discretize(data)
print(data)
#Equivale a coluna da classe, o objetivo eh achar os possiveis token nele
n_colums = len(data.columns)
lista = motherfucker(data, n_colums)
print(lista)
