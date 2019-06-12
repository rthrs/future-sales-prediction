import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# https://www.kaggle.com/dansbecker/your-first-machine-learning-model
# https://www.kaggle.com/learn/intro-to-machine-learning

training_set = pd.read_csv('./training_set.csv.gz')

# Create target object and call it y
y = training_set['item_cnt_day']

# Create X
# TODO !!! https://stackoverflow.com/questions/30384995/randomforestclassfier-fit-valueerror-could-not-convert-string-to-float
features = [
    'shop_id',
    'item_id',
    'item_price',
    'price_category', 
    'month_num', 
    'item_name', 
    'item_category_id',
    'name_1',
    'name_2',
    'name_3',
    'shop_name',
    'shop_city',
    'shop_type'
]
X = training_set[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)

val_predictions = dt_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
dt_model100 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
dt_model100.fit(train_X, train_y)

val_predictions = dt_model100.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
