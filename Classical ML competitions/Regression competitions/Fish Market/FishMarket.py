import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error


df = pd.read_csv("Fish.csv")
comparison = pd.DataFrame()

# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.corr())

# sns.heatmap(df.corr())
# plt.show()

# No need to delete columns

# print(df.isnull().sum())
# No null values

# print(df['Species'].value_counts())

# Check outliers


# plt.scatter(df['Length1'], df['Weight'])
# plt.show()


# plt.scatter(df['Length1'], df['Weight'])
# plt.show()


# plt.scatter(df['Length2'], df['Weight'])
# plt.show()


# plt.scatter(df['Length3'], df['Weight'])
# plt.show()


# plt.scatter(df['Width'], df['Weight'])
# plt.show()


lst = df[((df['Weight'] > 1500) & (df['Width'] < 7))].index

df.drop(lst, axis=0, inplace=True)
# print(df.shape)

# plt.scatter(df['Width'], df['Weight'])
# plt.show()


# plt.scatter(df['Height'], df['Weight'])
# plt.show()

# print(df.info())

target = pd.DataFrame(df['Weight'])
df.drop(['Weight'], axis=1, inplace=True)

df = pd.get_dummies(df)


# Splitting the data set into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.2, random_state=0)

comparison['Real Weight'] = y_test['Weight']

# print(y_test)
# print(df.head())
# print(df.info())

"""
Decision Tree
"""

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)
predictions_d_tree = dtr.predict(X_test)

print("Decision Tree Mean Squared error:",
      mean_squared_error(y_test, predictions_d_tree))
print("Accuracy:", dtr.score(X_test, y_test))
print("---------------------------------------------")


"""
Random Forest
"""

rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
rfr.fit(X_train, y_train)
predictions_r_forest = rfr.predict(X_test)

print("Random Forest Mean Squared error:",
      mean_squared_error(y_test, predictions_r_forest))
print("Accuracy:", rfr.score(X_test, y_test))
print("---------------------------------------------")


"""
Linear Regression
"""

lrr = LinearRegression()
lrr.fit(X_train, y_train)
predictions_l_reg = lrr.predict(X_test)

print("Linear Regreesion Mean Squared error:",
      mean_squared_error(y_test, predictions_l_reg))
print("Accuracy:", lrr.score(X_test, y_test))
print("---------------------------------------------")


# Random forest is the best
comparison['Predicted Weight'] = predictions_r_forest
comparison.to_csv("Results.csv", index=False)
