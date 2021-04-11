import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pasta.augment import inline
import pickle

mpl.style.use('ggplot')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

car = pd.read_csv('data/dataset.csv')
car = car.replace(r'^\s*$', np.nan, regex=True)
car.isna().sum()
car = car.dropna()
car = car.reset_index(drop=True)

car = car[car['price'] < 175000]
car = car[car['year'] >= 1990]
car = car[car['year'] < 2021]

plt.subplots(figsize=(20, 10))
ax = sns.boxplot(x='manufacturer', y='price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha='right')

plt.subplots(figsize=(20, 10))
ax = sns.barplot(x='year', y='price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha='right')

plt.subplots(figsize=(20, 10))
ax = sns.boxplot(x='manufacturer', y='year', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha='right')

plt.subplots(figsize=(20, 10))
ax = sns.scatterplot(x='odometer', y='price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

plt.xlim(0, 600000)
sns.scatterplot(x="odometer", y="price", data=car)
plt.show()

car = car[car['year'] <= 2011]

X = car[['model', 'manufacturer', 'year', 'odometer']]
y = car['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(X[['model', 'manufacturer']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['model', 'manufacturer']),
    remainder='passthrough')

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2_score(y_test, y_pred)

print("Estimated price of car:c  ")
print(pipe.predict(pd.DataFrame(columns=X_test.columns,
                                data=np.array(['tundra', 'toyota', 2009, 226999]).reshape(1, 4))))

pickle.dump(pipe, open('LinearRegressionModel-1.pkl', 'wb'))

# Model 2
car = pd.read_csv('data/dataset.csv')
car = car.replace(r'^\s*$', np.nan, regex=True)
car.isna().sum()
car = car.dropna()
car = car.reset_index(drop=True)
car = car[car['price'] < 175000]
car = car[car['year'] <= 2020]

X = car[['model', 'manufacturer', 'year', 'odometer']]
y = car['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(X[['model', 'manufacturer']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['model', 'manufacturer']),
    remainder='passthrough')

scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2_score(y_test, y_pred)

print(pipe.predict(pd.DataFrame(columns=X_test.columns,
                                data=np.array(['tundra', 'toyota', 2019, 226999]).reshape(1, 4))))

pickle.dump(pipe, open('LinearRegressionModel-2.pkl', 'wb'))
