from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

shuffled = df.sample(frac=1, random_state=17).reset_index(drop=True)

train_end = int(.8 * len(df))
valid_end =len(df)

train = shuffled.iloc[:train_end]
valid = shuffled.iloc[train_end:valid_end]

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler(random_state=17)
        x_scaled, y = ros.fit_resample(x_scaled, y)
    
    data  = np.hstack((x_scaled, y.reshape(-1, 1)))
    return x_scaled, y, data

x_train, y_train, train = scale_dataset(train, oversample=True)
x_valid, y_valid, valid = scale_dataset(valid, oversample=False)
