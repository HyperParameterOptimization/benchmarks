from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation as cval
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import classification_report

from time import time
from operator import itemgetter

import numpy as np
from scipy.stats import randint as sp_randint


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
# Initialize the data        

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = cval.train_test_split(digits.data, digits.target, test_size=0.3, random_state=0)

# Logistic regression without hyperparameter optimization

logit = LogisticRegression()

start = time()
logit.fit(X_train, y_train)
#logit.score(X_test, y_test)

print("============= Logistic Regression =============")
print("Logistic regression took %.2f seconds."
      % (time() - start))
print
print(classification_report(y_test, logit.predict(X_test)))

# Define parameter space to search over
# Note: GridSearch uses ParameterGrid method which does NOT interpolate between values.
# See grid_search.ParameterSampler for a random distribution sampler.
# Note: Use logit.get_params() to see default params.


"""
param_dist = {"C":[0.1,2],
              "intercept_scaling":[.02,2],
              "max_iter":[50],
              "tol":[0.00001,0.001]
             }
"""

C_param = list(np.linspace(0.1,2,5))
iter_num = [100]
inter_scale = list(np.linspace(0.1,1,5))

param_dist = {"C":[.5,1,1.5],
              "intercept_scaling":inter_scale,
              "max_iter":iter_num,
              "tol":[.0001],
              "penalty":['l1'],
              "class_weight":[None]
             }
# Initialize grid search

#grid_search = GridSearchCV(logit, param_dist)
grid_search = GridSearchCV(logit, param_grid=param_dist)

# Run grid search and check results
start = time()
grid_search.fit(X_train, y_train)

print("============= Grid Search =============")
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)

print("Best parameters set found on development set:")
print
print(grid_search.best_params_)
print
print("Grid scores on development set:")
print
for params, mean_score, scores in grid_search.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
print
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print
y_true, y_pred = y_test, grid_search.predict(X_test)
print(classification_report(y_true, y_pred))
print

# Random Search

rand_C_param = sp_randint(1,3)

rand_param_dist = {"C":sp_randint(1,3),           # Want range(0.5,1.5)
              "intercept_scaling":inter_scale,
              "max_iter":iter_num,
              "tol":[0.001],
              "penalty":['l1'],
              "class_weight":[None]
             }

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(logit, param_distributions=rand_param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
print("============= Random Search =============")
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
