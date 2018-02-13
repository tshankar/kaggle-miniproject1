import numpy as np
from sklearn import svm, grid_search

with open()
def svc_param_selection(X, y, nfolds):
    gammas = [0.001, 0.01, 0.1, 1]
    Cs = [0.001, 0.01, 0.1, 1, 10]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
