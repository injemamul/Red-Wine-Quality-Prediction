import numpy as np
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('winequality-red.csv', sep=';')

# Create Classification version of target variable
df['GoodQuality'] = [1 if x>=7 else 0 for x in df['quality']]

# Separete feature variables and target variable
X = df.drop(['quality', 'GoodQuality'], axis=1)
y = df['GoodQuality']

# Normalize Feature Variable
X = StandardScaler().fit_transform(X)

# Splitting Data into Train Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

# Creating RandomForest Classifier Model
rf_clf = RandomForestClassifier(n_estimators = 200, random_state=1)
rf_clf.fit(X_train, y_train)

# Prediction
rf_pred = rf_clf.predict(X_test)

# calculating the training and testing accuracies
print("Training accuracy :", rf_clf.score(X_train, y_train))
print("Testing accuracy :", rf_clf.score(X_test, y_test))

# Open a file where you want to store the data
file=open('random_forest_clf_model.pkl', 'wb')

# dumb information to that file
pickle.dump(rf_clf, file)

print("RandomForest Classification Model is being Exported")