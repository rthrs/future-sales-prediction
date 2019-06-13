import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def downcast_dtypes(df):
    """
    Downcast data to save memory space
    """
    float_cols = [c for c in df if df[c].dtype in ["float64", "float32"]]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

def encode_features(df, columns):
    for column in columns:
        if df[column].dtype == type(object):
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

def rmse(predictions, values):
    return mean_squared_error(predictions, values)**0.5

def save_submission(name, output):
    output.to_csv(f'{name}.csv.gz', index=False)

def save_submission2(name, test_data, test_preds):
    output = pd.DataFrame({'ID': test_data['ID'], 'item_cnt_month': test_preds})
    output.to_csv(f'{name}.csv.gz', index=False)
