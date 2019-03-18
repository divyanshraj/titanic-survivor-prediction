import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.nan)
Train_dataset = pd.read_csv('train.csv')
Test_dataset = pd.read_csv('test.csv')
X_train = Train_dataset.iloc[:, [2, 4,5]].values
y_train = Train_dataset.iloc[:, 1].values
X_test = Test_dataset.iloc[:, [1, 3,4]].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [0,2]])
X_train[:, [0,2]] = imputer.transform(X_train[:, [0,2]])
X_test[:, [0,2]] = imputer.transform(X_test[:, [0,2]])

from sklearn.preprocessing import LabelEncoder
labelencoder_X_train = LabelEncoder()
X_train[:, 1] = labelencoder_X_train.fit_transform(X_train[:, 1])
labelencoder_X_test = LabelEncoder()
X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])
labelencoder_y_train = LabelEncoder()
y_train = labelencoder_y_train.fit_transform(y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.cross_model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1, 10, 100, 1000]},
              ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


submission = Test_dataset.copy()
submission['Survived'] = y_pred
submission.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)

