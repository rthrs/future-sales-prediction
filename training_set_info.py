import pandas as pd

training_set = pd.read_csv('./training_set.csv.gz')

print("----------Top-5- Record----------")
print(training_set.head(5))
print("-----------Information-----------")
print(training_set.info())
print("-----------Data Types-----------")
print(training_set.dtypes)
print("----------Missing value-----------")
print(training_set.isnull().sum())
print("----------Null value-----------")
print(training_set.isna().sum())
print("----------Shape of Data----------")
print(training_set.shape)

train = pd.read_csv('./input/sales_train.csv.gz')
test = pd.read_csv('./input/test.csv.gz')

print('Number of duplicates:', len(training_set[training_set.duplicated()]))
print(len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test))

print(len(list(set(test.shop_id) - set(test.shop_id).intersection(set(train.shop_id)))), len(list(set(test.shop_id))), len(test))
print(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id))))
