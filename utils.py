
import numpy as np


def softmax(x):
    """
    x: [number fo observation, number of category]
    """
    max_x = np.max(x, axis=1, keepdims=True)
    p = np.exp(x - max_x)
    return p / np.sum(p, axis=1).reshape(-1,1)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(p):
    """
    take input as tanh(x)
    """
    return 1.0 - p ** 2



import sklearn
import sklearn.datasets

def load_data():
    N = 200

    gq = sklearn.datasets.make_gaussian_quantiles(
        mean=None, cov=0.7,
        n_samples=N, n_features=2,
        n_classes=2, shuffle=True,
        random_state=None,
    )
    
    return gq