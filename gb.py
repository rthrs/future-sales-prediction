import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_pickle('./training_set2.pkl')

X_train = train[train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = train[train.date_block_num < 33]['item_cnt_month']

X_test = train[train.date_block_num == 34].drop(['item_cnt_month'], axis=1)

print('Fitting...')
model = GradientBoostingRegressor(verbose=3)
model.fit(X_train, Y_train)

print('Predicting...')
Y_test = model.predict(X_test)

submission = pd.DataFrame({
    "ID": range(214200), 
    "item_cnt_month": Y_test
})

print('Saving file...')
submission.to_csv('submission_gb.csv', index=False)

