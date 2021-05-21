

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# example of summarizing the number of missing values for each variable
from pandas import read_csv
from numpy import nan
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load the dataset
dataset = read_csv('Book2.csv', header=None)
# count the number of missing values for each column
values = dataset.values


X = values[:,0:21]
y = values[:,21]
# define the model
# define the model

# Fit regression model
#svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)


lw = 2

svrs = [svr_lin]#, svr_rbf, svr_poly]
kernel_label = ['Linear']#'RBF, 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    regressor = svr.fit(X, y)
    xPred = regressor.predict(X)
    score = regressor.score(X, y)
    print('Accuracy: %.3f' % score)
    axes[ix].plot(y, xPred, color=model_color[ix])

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)