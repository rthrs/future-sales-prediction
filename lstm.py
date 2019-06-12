import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler

from math import ceil

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

# Read data
train = pd.read_csv('./input/sales_train.csv')
test = pd.read_csv('./input/test.csv')
submission = pd.read_csv('./input/sample_submission.csv')
items = pd.read_csv('./input/items.csv')
item_cats = pd.read_csv('./input/item_categories.csv')
shops = pd.read_csv('./input/shops.csv')

# Validate data
test_shops = test.shop_id.unique()
train = train[train.shop_id.isin(test_shops)]
test_items = test.item_id.unique()
train = train[train.item_id.isin(test_items)]

# Constants depenedent on read data
MAX_BLOCK_NUM = train.date_block_num.max()
MAX_ITEM = len(test_items)
MAX_CAT = len(item_cats)
MAX_YEAR = 3
MAX_MONTH = 4 # 7 8 9 10
MAX_SHOP = len(test_shops)

