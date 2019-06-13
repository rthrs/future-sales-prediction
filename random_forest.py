import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from utils import downcast_dtypes, encode_features, rmse, save_submission

# https://www.kaggle.com/dansbecker/your-first-machine-learning-model
# https://www.kaggle.com/learn/intro-to-machine-learning

training_set = downcast_dtypes(pd.read_csv('./training_set.csv.gz'))

# Create target object and call it y
train_y = training_set['item_cnt_day']

# Create X
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

encode_features(training_set, features)

train_X = training_set[features]

# TODO probably can add some various models to array, iterate and save different submission for them...
print('Training...')
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

# Predict test data
test = pd.read_csv('./input/test.csv.gz')

# Featurize known data
test_X = pd.merge(test, train_X, on=['shop_id', 'item_id'], how='inner')

# Set month_num to November
test_X['month_num'] = 11

# Predict values
print('Predicting...')
rf_val_predictions = rf_model.predict(test_X[features])

output = pd.DataFrame({'ID': test_X['ID'], 'item_cnt_month': rf_val_predictions})

# Set prediction of uknown pairs
submission_cols = ['ID', 'item_cnt_month']

output = pd.merge(test, output, on='ID', how='left')[submission_cols]
output.fillna(0.0, inplace=True)
print(output.shape)

# Have to output aggregated data... For now just do somethinf very stupid
# Multiplicate each daily sale by number of days in November and take mean form the pair...
output['item_cnt_month'] = output['item_cnt_month'].apply(lambda x: x * 30)
output = output.groupby('ID').agg({'ID': 'first', 'item_cnt_month': 'mean'})

print('Saving file...')

save_submission('random_forest_submission', output)
