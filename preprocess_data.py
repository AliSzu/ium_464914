import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split(data):
    forest_train, forest_test = train_test_split(data, test_size=0.2, random_state=1)
    forest_train, forest_val = train_test_split(forest_train, test_size=0.25, random_state=1)
    return forest_train, forest_test, forest_val

def normalization(data):
    scaler = StandardScaler()
    columns_to_normalize = data.columns[~data.columns.str.startswith('Soil_Type')]
    columns_to_normalize = columns_to_normalize.to_list()
    columns_to_normalize.remove('Cover_Type')
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    return data

def preprocessing(data):
    #shuffle
    data = data.sample(frac = 1)
    return data

data = pd.read_csv("covtype.csv")
forest_train, forest_test, forest_val = split(data)

forest_train = preprocessing(forest_train)
forest_test = preprocessing(forest_test)
forest_val = preprocessing(forest_val)

forest_train = normalization(forest_train)
forest_test = normalization(forest_test)
forest_val = normalization(forest_val)

forest_train.to_csv('forest_train.csv', encoding='utf-8', index=False)
forest_test.to_csv('forest_test.csv', encoding='utf-8', index=False)
forest_val.to_csv('forest_val.csv', encoding='utf-8', index=False)


