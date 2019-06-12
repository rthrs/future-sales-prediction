import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def downcast_dtypes(df):
    """
    Downcast data to save memory space
    """
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

def encode_features(df, columns):
    for column in columns:
        if df[column].dtype == type(object):
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

def rmse(predictions, values):
    return mean_squared_error(predictions, values)**0.5

# Change string to floats - didn't work
# enc = OneHotEncoder(handle_unknown='ignore')
# str_cols = training_set.columns[training_set.dtypes.eq('object')]
# training_set[str_cols] = training_set[str_cols].apply(pd.to_numeric, errors='coerce')
