import numpy as np
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv('winequality-red.csv', sep=';')

# Create Classification version of target variable
df['GoodQuality'] = [1 if x>=7 else 0 for x in df['quality']]

# Separete feature variables and target variable
X = df.drop(['quality', 'GoodQuality'], axis=1)
y = df['GoodQuality']


# Defining and fitting Smote to perform upSampling
sm = SMOTE()
X, y = sm.fit_sample(X, y)

# Normalize Feature Variable
X = StandardScaler().fit_transform(X)

# Splitting Data into Train Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)

# Defining and fitting Random Forest classifier with our best parameter
model_rf = RandomForestClassifier(n_estimators=200,
                                        criterion='gini',
                                        max_depth=8,
                                        max_features='auto')
model_rf.fit(X_train, y_train)

# Prediction
pred_y = model_rf.predict(X_test)

# calculating the training and testing accuracies
print("Training accuracy :", model_rf.score(X_train, y_train))
print("Testing accuracy :", model_rf.score(X_test, y_test))
print("Classification report :", classification_report(y_test, pred_y))

# Open a file where you want to store the data
file = open('random_forest_clf_model.pkl', 'wb')

# dumb information to that file
pickle.dump(model_rf, file)

print("RandomForest Classification Model is being Exported")
