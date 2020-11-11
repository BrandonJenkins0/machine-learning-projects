# Loading modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.metrics import log_loss

# Reading in the data
train_data = pd.read_csv('Biological Response/data/train.csv')
test_data = pd.read_csv('Biological Response/data/test.csv')

# Splitting data into test and train set
X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['Activity'], axis=1), train_data['Activity'],
                                                    random_state=1, stratify=train_data['Activity'])


# Functions to smooth out the process
def model_performance(pipe):
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f'Average: {np.mean(scores)} \nSt. Dev {np.std(scores)}')


# Creating quick baseline model
pipe1 = Pipeline([('log', LogisticRegressionCV(penalty="none", max_iter=2500))])

# Using cross validation to estimate model performance
model_performance(pipe1)

# Lets try reducing the number of features we are using to reduce dimensionality.
# Try L1 regularization to choose features
pipe2 = Pipeline([('log1', LogisticRegressionCV(penalty='l1', solver='saga', max_iter=3000, cv=5))])
pipe2.fit(X_train, y_train)
model_performance(pipe2)
log = pipe2.named_steps['log1']
log.best_estimator
coef = pd.Series(log.coef_[0])
top50_features = (coef[coef != 0]**2).sort_values(ascending=False)[:50]

top50_features

# Refitting a logistic regression model using the top 50 features. L2 regularization. Gonna tune a bit as well.
X_train_top_50_l1 = X_train.iloc[:, top50_features.index]
log2 = LogisticRegression(max_iter=1000)
model_performance(log2)

tuning_grid = {'C': np.arange(.5, 1.6, .1)}
tuned_log = GridSearchCV(log2, tuning_grid, n_jobs=-1)
tuned_log.fit(X_train_top_50_l1, y_train)

tuned_log.best_params_
tuned_log.scorer_
results = pd.DataFrame(tuned_log.cv_results_)

# Refit with all data.
tuned_log.refit

# Test data prediction using best logistic regression l1 method
X_test_top_50_l1 = test_data.iloc[:, top50_features.index]
predictions = tuned_log.predict_proba(X_test_top_50_l1)

pd.DataFrame({'MoleculeID': test_data.index + 1, 'PredictedProbability': predictions[:,1]})\
    .to_csv('Biological Response/data/Logl1_predictions.csv', index=False)

# Let's try fitting just a baseline random forest model. See if it performs better than my tuned logistic regression mod
rf = RandomForestClassifier(random_state=1, n_jobs=-1)
model_performance(rf)

param_grid = {"n_estimators": [1900],
              "max_depth": [None],
              "max_features": ['auto']}
cv_clf = GridSearchCV(rf, param_grid, n_jobs=-1, cv=5)
cv_clf.fit(X_train, y_train)
best = cv_clf.best_estimator_
best.feature_importances_

feat_import = pd.DataFrame({'Feature': X_train.columns, 'Importance': best.feature_importances_})
top50_features_rf = feat_import.sort_values("Importance", ascending=False).index[:50]

X_train_top_50_rf = X_train.iloc[:, top50_features_rf]

cv_clf2 = GridSearchCV(rf, param_grid=param_grid, cv=3, scoring='neg_log_loss')
cv_clf2.fit(X_train_top_50_rf, y_train)

cv_clf2.best_score_ * -1

# Test data prediction using best logistic regression l1 method
X_test_top_50_rf = test_data.iloc[:, top50_features_rf]
predictions = cv_clf2.predict_proba(X_test_top_50_rf)

pd.DataFrame({'MoleculeID': test_data.index + 1, 'PredictedProbability': predictions[:,1]})\
    .to_csv('Biological Response/data/rf_predictions.csv', index=False)

# PCA with some random forest action
pipe3 = Pipeline(
    [('pca', KernelPCA(n_components=10)),
     ('rf', cv_clf2)]
)

scores = cross_val_score(pipe3, X_train, y_train, cv=5, scoring='neg_log_loss')

pipe3.fit(X_train, y_train)
predictions = pipe3.predict_proba(X_train)
log_loss(y_train, predictions)