
# coding: utf-8

# # Classifier for Kaggle popcorn competition
# 
# Uses logistic regression to try and predict sentiment from a bag-of-words summary of each review.

# In[2]:

import pandas as pd
import numpy as np
import scipy.optimize as opt
# Expit is just the sigmoid function, written in C
from scipy.special import expit

data = pd.read_csv('even_better_features.csv', dtype='float32')

ys = data.iloc[:, 0]
data = data.iloc[:, 1:]


# In[3]:

data['ones_column'] = np.ones_like(ys)


# In[4]:

from functools import wraps
from timeit import default_timer

def timed(f):
    """Decorator to measure the runtime of a function and print it to
    stdout."""
    if hasattr(f, 'func_name'):
        name = f.func_name
    elif hasattr(f, 'im_func'):
        name = f.im_func.func_name
    elif hasattr(f, '__name__'):
        name = f.__name__
    else:
        name = '<unknown>'

    @wraps(f)
    def inner(*args, **kwargs):
        start = default_timer()
        rv = f(*args, **kwargs)
        taken = default_timer() - start
        print("Call to {} took {}s".format(name, taken))
        return rv

    return inner


# In[5]:

def safe_sigmoid(x):
    """Computes sigmoid function, but in the numerically stable way used by Sklearn"""
    gt_zero = x > 0
    gt_zero_x = x[gt_zero]
    not_gt_zero_x = x[~gt_zero]
    rv = np.zeros_like(x)
    rv[gt_zero] = 1.0 / (1 + np.exp(-gt_zero_x))
    rv[~gt_zero] = np.exp(not_gt_zero_x) / (np.exp(not_gt_zero_x) + 1)
    return rv

def safe_log_sigmoid(x):
    """Compute log(sigmoid(x)) safely"""
    rv = np.zeros(len(x))
    mask = x > 0
    rv[mask] = -np.log(1 + np.exp(-x[mask]))
    rv[~mask] = x[~mask] - np.log(1 + np.exp(x[~mask]))
    return rv

def safe_log_one_minus_sigmoid(x):
    """Compute log(1 - sigmoid(x)) safely"""
    rv = np.zeros(len(x))
    mask = x > 0
    rv[mask] = -x[mask] - np.log(1 + np.exp(-x[mask]))
    rv[~mask] = -np.log(np.exp(x[~mask]) + 1)
    return rv

@timed
def logistic_gradient(weights, features, labels):
    """Compute the gradient of the cross-entropy error for logistic regression"""
    predictions = expit(np.dot(features, weights))
    error = predictions - labels
    # We divide by label.size to try and keep numerical error under control
    return np.dot(error, features) / labels.size

@timed
def cross_entropy_error(weights, features, labels):
    """Cross-entropy error for logistic regression"""
    dots = np.dot(features, weights)
    rv = labels*safe_log_sigmoid(dots) + (1-labels)*safe_log_one_minus_sigmoid(dots)
    return -np.mean(rv)

def logistic_predict(weights, features):
    """Predict labels for the given features using the given weights"""
    probs = expit(np.dot(features, weights))
    rv = np.zeros(len(features))
    rv[probs > 0.5] = 1
    return rv

def train_logistic_regression(labels, features):
    """Find a weight vector for logistic regression using the supplied
    labels and features."""
    np_labels = np.array(labels)
    
    initial_w = 0.1 * np.random.random(features.shape[1])
    assert initial_w.ndim == 1
    
    result = opt.fmin_bfgs(
        cross_entropy_error, 
        initial_w, 
        fprime=logistic_gradient,
        args=(features, np_labels),
        maxiter=100
    )
    
    return result


# In[10]:

print(cross_entropy_error(np.random.random(data.shape[1]), data, ys))
print(logistic_gradient(np.random.random(data.shape[1]), data, ys))


# In[ ]:

ws = train_logistic_regression(ys, data)
predictions = logistic_predict(ws, data)
print("Incorrect guesses:")
print(np.sum(predictions != ys))
print("Correct guesses:")
print(np.sum(predictions == ys))

# In[ ]:



