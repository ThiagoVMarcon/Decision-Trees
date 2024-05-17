import numpy as np
import pandas as pd
 
class Node():
    # constructor
    def __init__(self):
        self.feature = None #Name of the feature in the node
        self.childs_name = [] #Childrens names
        self.childs = [] #Childrens list
        self.value = None #count of the most occurances of the featured class attribute in a leaf
        self.class_name = None #label of the most occurances of the featured class attribute in a leaf

#discretizes the numerical features of the DataFrame
def discretize(data): #Verifica se o dado em uma dada coluna eh numerico, se for o discretiza
        num_of_columns = data.shape[1]
        for column in range(num_of_columns-1):
            try:
                float(data.iloc[0,column])
                #checks if value is False or True
                if(data.iloc[0,column] == False or data.iloc[0,column] == True): #if yes then break, python treats these booleans as floats
                    continue
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
        return labels

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

def entropy(data):
    lista = []
    max = [-float("inf"), -1] #Primeira valor max, segunda pos do valor max
    unique_goal_name = data.columns[len(data.columns)-1]
    goal_values = data[unique_goal_name].unique()
    for i in range(0,data.shape[1] - 1):
        column_name = data.columns[i]
        #print(column_name)
        x = calc_info_gain(column_name, data, unique_goal_name, goal_values)
        if(x> max[0]):
            max[1] = i
            max[0] = x
        lista.append(x)
    if(max[0] == 0.0): #se entropia for 0 retorna valor espacial de handling
        return -1
    return(data.columns[max[1]])

def build_Tree(data):
    root = Node()
    unique_goal_name = data.columns[len(data.columns)-1]
    if(data[unique_goal_name].nunique() == 1 or entropy(data) == -1):###### MUDAR MUDAR
        root.feature = None
        goal_values = data[unique_goal_name].unique()
        list_goal_values = []
        for c in goal_values:
            label_count = data[data[unique_goal_name] == c].shape[0]
            list_goal_values.append([c,label_count])
        maior_par = max(list_goal_values, key=lambda x: x[1])
        root.value = maior_par[1]
        root.class_name = maior_par[0]
        return root
    best_feature = entropy(data)
    root.feature = best_feature
    root.childs_name = data[best_feature].unique()
    for current in root.childs_name: #current eh o atributo ex: Yes, No e etc
        new_data = data[data[best_feature] == current]
        child_node = build_Tree(new_data)
        root.childs.append(child_node)
    return root

def print_Tree(root, indentation=''):
    
    # Recursively traverse the tree and generate the output string
    if root.feature is None:
        print(str(root.class_name) + " total:" + str(root.value), end="")
    if len(root.childs) == 0:
        return 
    counter = 0 #PORCO
    for child in root.childs:
        print()
        print(indentation + '<' + str(root.feature) + '>')
        print(indentation + ' ' + str(root.childs_name[counter]) + ":", end=" ")
        print_Tree(child, indentation + '   ')
        counter += 1

def intervalo_para_min_max(intervalo):
    # Checar exclusividade no máximo
    max_exclusivo = intervalo[-1] == ')'
    
    # Remover os colchetes e parênteses
    intervalo = intervalo.strip('[]()')
    
    # Dividir a string pelo hífen
    min_val, max_val = intervalo.split('-')
    
    # Converter para float
    min_val = float(min_val)
    max_val = float(max_val)
    
    return min_val, max_val

def valor_em_intervalo(min, max,valor):  
    if min <= valor <= max:
        return True
    return False

def predict(root, new_data, original_data):
    if not(root.value is None):
        return root.class_name
    counter = 0 ###PORCOOOOOOO
    new_feature_name = new_data[root.feature][0]
    for i in root.childs_name:
        try:
            float(new_feature_name)
            if((new_feature_name== False or new_feature_name == True)):
                pass
            discretized_intervals = original_data[root.feature].unique()
            for i2 in discretized_intervals:
                min, max = intervalo_para_min_max(i2)
                if(valor_em_intervalo(min, max,new_feature_name)):
                    new_data[root.feature] = i2
                    new_feature_name = new_data[root.feature][0]
                    break
        except ValueError:
            pass
        if(new_feature_name==i):
            return predict(root.childs[counter], new_data, original_data)
        counter += 1

with open("Datasets/iris.csv") as x:
    ncols = len(x.readline().split(','))
data = pd.read_csv("Datasets/iris.csv", usecols=range(1,ncols))
with open("Datasets/iris_teste.csv") as x:
    ncols = len(x.readline().split(','))
data_teste = pd.read_csv("Datasets/iris_teste.csv", usecols=range(1,ncols))
lista_labels =discretize(data)
if (data.isnull().values.any()):
    data = data.fillna('Nan')
print(data)
#Equivale a coluna da classe, o objetivo eh achar os possiveis token nele
#name = entropy(data, len(data.columns))
#print(name)
teste = build_Tree(data)
print_Tree(teste)
print()
y = predict(teste, data_teste, data)
print(y)


