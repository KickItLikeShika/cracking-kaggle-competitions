import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

result = pd.DataFrame(test['PassengerId'])

# print(train.head())
# print(test.head())

# print(train.info())
sns.heatmap(train.isnull())
plt.show()

train.drop(['Name', 'PassengerId', 'Ticket',
            'Cabin', 'Fare'], axis=1, inplace=True
           )

train['Age'] = train['Age'].fillna(
    train['Age'].mean()
)

train['Embarked'] = train['Embarked'].fillna(
    train['Embarked'].mode()[0]
)

# sns.heatmap(test.isnull())
# plt.show()

test.drop(['Name', 'PassengerId', 'Ticket',
           'Cabin', 'Fare'], axis=1, inplace=True
          )

test['Age'] = test['Age'].fillna(
    test['Age'].mean()
)

test['Embarked'] = test['Embarked'].fillna(
    test['Embarked'].mode()[0]
)

target = pd.DataFrame(train['Survived'])
train.drop(['Survived'], axis=1, inplace=True)

# get dummies does what's one hot encoder does
# without any need to column transformer

# g = sns.FacetGrid(train, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# # Explore Sex vs Survived
# g = sns.barplot(x="Sex", y="Survived", data=train)
# g = g.set_ylabel("Survival Probability")


# # Explore Pclass vs Survived
# g = sns.catplot(x="Pclass", y="Survived", data=train, kind="bar", height=6,
#                 palette="muted")
# g.despine(left=True)
# g = g.set_ylabels("survival probability")


# print(train.head())
# print(train.info())
# print(test.head())
# print(test.info())
train = pd.get_dummies(train)
test = pd.get_dummies(test)
# print(train.head())
# print(train.info())
# print(test.head())
# print(test.info())
# print(target.info())

rfc = RandomForestClassifier(n_estimators=100,
                             max_features='auto',
                             criterion='entropy',
                             max_depth=10)

rfc.fit(train, target.values.ravel())


y_pred = rfc.predict(test)

temp = pd.DataFrame(y_pred)

result['Survived'] = y_pred

print(result)

result.to_csv("submission.csv", index=False)

print("Results converted to csv successfully")
