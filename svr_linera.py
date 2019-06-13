import pandas as pd

from sklearn.svm import SVR

train = pd.read_pickle('./training_set2.pkl')

X_train = train[train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = train[train.date_block_num < 33]['item_cnt_month']

X_test = train[train.date_block_num == 34].drop(['item_cnt_month'], axis=1)

print('Fitting...')
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model.fit(X_train, Y_train)

print('Predicting...')
Y_test = model.predict(X_test)

submission = pd.DataFrame({
    "ID": range(214200), 
    "item_cnt_month": Y_test
})

print('Saving file...')
submission.to_csv('submission_svr_rbf.csv', index=False)
