import numpy as np
import pandas as pd

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv("Datasets/iris.csv", skiprows=1, header=None, names=col_names)

class Node():
    # constructor
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTreeClassifier():
    # constructor
    def __init__(self, min_samples_split=2, max_depth=2):
        # initialize the root of the tree 
        self.root = None
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    # recursive function to build the tree
    def build_tree(self, dataset, curr_depth=0):
        # assigns the values of the dataset's attributes to X and the labels to Y
        X, Y = dataset[:,:-1], dataset[:,-1]
        # calculate the shape of the array X the samples (rows) and the features (columns)
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    # function to find the best split (based on information gain)
    def get_best_split(self, dataset, num_samples, num_features):
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split
    
    # function to split the data (left and right)
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right   
    
    # function to compute information gain (based on entropy or gini index)
    def information_gain(self, parent, l_child, r_child, mode="entropy"):        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    # function to compute entropy
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    # function to compute gini index
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    # function to compute leaf node
    def calculate_leaf_value(self, Y):        
        Y = list(Y)
        return max(Y, key=Y.count)

    # function to print the tree
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)   
    
    # discretizes the numerical attributes of the DataFrame
    def discretize(self, data):
        num_of_columns = data.shape[1]
        for column in range(1, num_of_columns):
            try:
                data.iloc[:, column] = data.iloc[:, column].astype(float)
                self.do_Discretize(column, data)
            except ValueError:
                pass
    
    # discretizes a single numerical column of the DataFrame (helper of discretize)
    def do_Discretize(self, column, data):
        column_name = data.columns[column]
        std_dev = np.std(data[column_name])
        num_points = len(data[column_name])
        bin_width = 3.5 * std_dev / (num_points ** (1/3))
        data_range = data[column_name].max() - data[column_name].min()
        num_bins = int(np.ceil(data_range / bin_width))
        labels = [f'[{data[column_name].min() + i * bin_width:.2f}-{data[column_name].min() + (i + 1) * bin_width:.2f})' for i in range(num_bins)]
        data[column_name] = pd.qcut(data[column_name], q=num_bins, labels=labels)

    # function to train the tree
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    # function to predict new dataset
    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    # function to predict a single data point
    def make_prediction(self, x, tree):        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# Example usage
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

classifier = DecisionTreeClassifier(min_samples_split=5, max_depth=5)
classifier.discretize(pd.DataFrame(X))
classifier.fit(X, Y)
classifier.print_tree()
