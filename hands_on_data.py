import pandas as pd
import numpy as np

# Raw csv data

sale_train = pd.read_csv('./input/sales_train.csv')
test = pd.read_csv('./input/test.csv')

# Downcast data to save memory space

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

sale_train = downcast_dtypes(sale_train)

# Data basic infos
print(sale_train.info())

print("----------Top-5- Record----------")
print(sale_train.head(5))
print("-----------Information-----------")
print(sale_train.info())
print("-----------Data Types-----------")
print(sale_train.dtypes)
print("----------Missing value-----------")
print(sale_train.isnull().sum())
print("----------Null value-----------")
print(sale_train.isna().sum())
print("----------Shape of Data----------")
print(sale_train.shape)

print('Number of duplicates:', len(sale_train[sale_train.duplicated()]))

# Some data insights
sales_by_item_id = sale_train.pivot_table(index=['item_id'],values=['item_cnt_day'], 
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'

print(sales_by_item_id)

outdated_items = sales_by_item_id[sales_by_item_id.loc[:,'27':].sum(axis=1)==0]

print('Outdated items in test set:', len(test[test['item_id'].isin(outdated_items['item_id'])]))
print()

#  3 GROUPS OF DATA!!!
good_sales = test.merge(sale_train, on=['item_id','shop_id'], how='left').dropna()

good_pairs = test[test['ID'].isin(good_sales['ID'])]
no_data_items = test[~(test['item_id'].isin(sale_train['item_id']))]

print('All items in test set', len(test['item_id']))
print('1. Number of good pairs:', len(good_pairs))
print('2. No Data Items:', len(no_data_items))
print('3. Only Item_id Info:', len(test)-len(no_data_items)-len(good_pairs))
