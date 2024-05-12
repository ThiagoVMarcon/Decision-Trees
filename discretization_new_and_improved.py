import pandas as pd
import numpy as np

##### Para deixar correto trocar selfa -> self e dataf -> data para as duas funções 
def discretize(selfa, dataf):
    ############# DELETE START
    data = {
    "firstname": ["Sally", "Mary", "John", "Ashley"],
    "age": [50, 40, 30, 18],
    "sex": ["F","F","M","F"],
    "rnumber": [1,2,3,4]
}   
    ############ DELETE END
    df = pd.DataFrame(data)
    num_of_columns = df.shape[1]
    for column in range(num_of_columns):
        try:
            df.iloc[0,column] = float(df.iloc[0,column])
            do_Discretize(6, column, 5) #trocar 6 por self e 5 por data
            print(df.iloc[0,column])
            print(df.columns[column])
        except ValueError:
            pass
def do_Discretize(selfa, column, dataf):
    # Calculate the standard deviation and number of data points
    ############################## DELETE START
    data = {
    "firstname": ["Sally", "Mary", "John", "Ashley"],
    "age": [50, 40, 30, 18],
    "sex": ["F","F","M","F"],
    "rnumber": [1,2,3,4]
}   
    df = pd.DataFrame(data)
    ###############################DELETE END
    column_name = df.columns[column]
    std_dev = np.std(df[column_name])
    num_points = len(df[column_name])

    # Calculate the bin width using Scott's Rule
    bin_width = 3.5 * std_dev / (num_points ** (1/3))

    # Calculate the number of bins based on the bin width
    data_range = df[column_name].max() - df[column_name].min()
    num_bins = int(np.ceil(data_range / bin_width))

    #print("Number of bins according to Scott's Rule:", num_bins)
    
    # Generate labels as ranges for the values
    labels = [f'[{df[column_name].min() + i * bin_width:.2f}-{df[column_name].min() + (i + 1) * bin_width:.2f})' for i in range(num_bins)]
    df[column_name] = pd.qcut(df[column_name], q=num_bins, labels=labels)
    print(df)


discretize(5, 4)
print("Por hora os valores discretizados estao separados, quando feitos no mesmo dataset devem estar ambos sendo aplicados")
print("Nao fiz pois criar uma classe so pra isso ia ficaria muita bagunça")