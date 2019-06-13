# https://www.kaggle.com/dlarionov/feature-engineering-xgboost

"""
We'll generate matrix for each possible shop_id, item_id and date_block_num
up to 34th block, the one for witch sales should be predicted...
"""

import pandas as pd
import numpy as np

from operator import itemgetter
from itertools import product
from utils import downcast_dtypes, encode_features

import csv

# Raw csv data
train = pd.read_csv('./input/sales_train.csv.gz')
items = pd.read_csv('./input/items.csv')
categories = pd.read_csv('./input/item_categories.csv')
shops = pd.read_csv('./input/shops.csv')
test = pd.read_csv('./input/test.csv.gz').set_index('ID')

# Coerce data
train = downcast_dtypes(train)

# Remove outliers
train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]

# Repair data
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median

train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

def get_price_category(price):
    if (price < 2):
        return 0
    if (price < 10):
        return 1
    if (price < 20):
        return 2
    if (price < 100):
        return 3
    if (price < 500):
        return 4
    if (price < 1000):
        return 6
    if (price < 5000):
        return 7
    return 8

# Add items features
items['item_name_1'], items['item_name_2'] = items['item_name'].str.split('[', 1).str
items['item_name_1'], items['item_name_3'] = items['item_name_1'].str.split('(', 1).str

items['item_name_2'] = items['item_name_2'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items['item_name_3'] = items['item_name_3'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()

items = items.fillna('NO_DATA').replace(r'^\s*$', 'NO_DATA', regex=True)
encode_features(items, items.columns) # do categorical variables encoding....

# Add shops featues
shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()
shops['shop_city'] = shops['shop_name'].str.partition(' ')[0]
shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
encode_features(shops, shops.columns) # do categorical variables encoding....

# Add sales train features
train['price_category'] = train['item_price'].map(lambda x: get_price_category(x))
train['month_num'] = train['date_block_num'].map(lambda x: x % 12) # TODO potem ???


# Monthly sales
matrix = []
cols = ['date_block_num','shop_id','item_id']

for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)

group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
    .fillna(0)
    .astype(np.float16))

# Add test set as 34th month block
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month

# Merge datasets
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, categories, on=['item_category_id'], how='left')

# HERE you can add additional features, avg item count or price per specific city, shop type, etc...
# TODO...

# Add date features

matrix['month'] = matrix['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)

# Drop unused columns
matrix.drop(['item_name', 'item_category_name'], inplace=True, axis=1)

# Save dataset

matrix = downcast_dtypes(matrix)
matrix.to_pickle('training_set2.pkl')
# matrix.to_csv('training_set2.csv.gz', index=False)

# After that with xgboost we can do some feature engineering, but probably no time for that...

print(matrix.info())
print(matrix.head())
