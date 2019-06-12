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

print('Number of duplicates:', len(training_set[training_set.duplicated()]))