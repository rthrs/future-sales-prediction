import pandas as pd
import numpy as np

from operator import itemgetter
from utils import downcast_dtypes

import csv

# Raw csv data
sales_train = pd.read_csv('./input/sales_train.csv.gz')
items = pd.read_csv('./input/items.csv')
item_categories = pd.read_csv('./input/item_categories.csv')
shops = pd.read_csv('./input/shops.csv')
test = pd.read_csv('./input/test.csv.gz')

# Coerce data
sales_train = downcast_dtypes(sales_train)
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')

# Sales pairs for prediction which are available in provided training set
good_sales = test.merge(sales_train, on=['item_id','shop_id'], how='left').dropna()

#  3 GROUPS OF DATA!
good_pairs = test[test['ID'].isin(good_sales['ID'])]
no_data_items = test[~(test['item_id'].isin(sales_train['item_id']))]
# only_item_info = test[~test['ID'].isin((good_pairs + no_data_items)['ID'])] # TODO

# Prepare data on which we'll train model

# TODO predict values for not known items?

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
items['name_1'], items['name_2'] = items['item_name'].str.split('[', 1).str
items['name_1'], items['name_3'] = items['name_1'].str.split('(', 1).str

items['name_2'] = items['name_2'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items['name_3'] = items['name_3'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()

items = items.fillna('NO_DATA').replace(r'^\s*$', 'NO_DATA', regex=True)

# Add shops featues
shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()
shops['shop_city'] = shops['shop_name'].str.partition(' ')[0]
shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')


# Add sales train features
sales_train['price_category'] = sales_train['item_price'].map(lambda x: get_price_category(x))
sales_train['month_num'] = sales_train['date'].map(lambda x: x.month)

# Merge datasets
dataset = sales_train[sales_train['shop_id'].isin(good_pairs['shop_id']) & sales_train['item_id'].isin(good_pairs['item_id'])]
dataset = pd.merge(dataset, items, on='item_id')
dataset = pd.merge(dataset, shops, on='shop_id')
dataset = pd.merge(dataset, item_categories, on='item_category_id')

print(dataset.head())

# Save dataset
dataset.to_csv('training_set.csv.gz', index=False)
