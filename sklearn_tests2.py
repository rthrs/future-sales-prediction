import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from utils import downcast_dtypes, encode_features, rmse

# https://www.kaggle.com/dlarionov/feature-engineering-xgboost

train = pd.read_pickle('./training_set2.pkl')

X = train.drop(['item_cnt_month'], axis=1)
y = train['item_cnt_month']

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)

val_predictions = dt_model.predict(val_X)
val_rmse = rmse(val_y, val_predictions)
print("Validation RMSE when not specifying max_leaf_nodes: {}".format(val_rmse))
# Validation RMSE when not specifying max_leaf_nodes: 2.5498269967793057

# Using best value for max_leaf_nodes
dt_model100 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
dt_model100.fit(train_X, train_y)

val_predictions = dt_model100.predict(val_X)
val_rmse = rmse(val_predictions, val_y)
print("Validation RMSE for best value of max_leaf_nodes: {}".format(val_rmse))
# Validation RMSE for best value of max_leaf_nodes: 2.825573731965393

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1, verbose=3)
rf_model.fit(train_X, train_y)

val_predictions = rf_model.predict(val_X)
rf_val_rmse = rmse(val_predictions, val_y)
print("Validation RMSE for Random Forest Model: {}".format(rf_val_rmse))
# Validation RMSE for Random Forest Model: 2.238423610462634

model = LinearRegression()
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
rf_val_rmse = rmse(val_predictions, val_y)
print("Validation RMSE for Linear Regression: {}".format(rf_val_rmse))
# Validation RMSE for Linear Regression: 3.37160140437921

model = SVR(kernel='linear', verbose=True)
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
rf_val_rmse = rmse(val_predictions, val_y)
print("Validation RMSE for SVR rbf: {}".format(rf_val_rmse))

model = LogisticRegression(verbose=3)
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
rf_val_rmse = rmse(val_predictions, val_y)
print("Validation RMSE for SVR rbf: {}".format(rf_val_rmse))

model = GradientBoostingRegressor(verbose=3)
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
rf_val_rmse = rmse(val_predictions, val_y)
print("Validation RMSE for GradientBoostingRegressor: {}".format(rf_val_rmse))
