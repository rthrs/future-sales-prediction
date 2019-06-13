import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from utils import downcast_dtypes, encode_features, rmse, save_submission

# https://www.kaggle.com/dlarionov/feature-engineering-xgboost

train = pd.read_pickle('./training_set2.pkl')

X_train = train[train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = train[train.date_block_num < 33]['item_cnt_month']

X_test = train[train.date_block_num == 34].drop(['item_cnt_month'], axis=1)

print('Fitting...')
model = RandomForestRegressor(verbose=3)
model.fit(X_train, Y_train)

print('Predicting...')
Y_test = model.predict(X_test)

submission = pd.DataFrame({
    "ID": range(214200), 
    "item_cnt_month": Y_test
})

print('Saving file...')
submission.to_csv('submission_rf2.csv', index=False)
