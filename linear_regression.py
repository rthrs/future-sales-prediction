import pandas as pd
import numpy as np

# Raw csv data

training_set = pd.read_csv('./training_set.csv')
test = pd.read_csv('./input/test.csv')

# Downcast data to save memory space

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

training_set = downcast_dtypes(training_set)

# Split data to feature we'll train on and target variable to predict

X = training_set[['item_id', 'shop_id', '', '']]
y = training_set['']
