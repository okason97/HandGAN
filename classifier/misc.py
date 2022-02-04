import numpy as np
from math import exp

def sigmoid(x):
    return 1/(1+exp(-x))

def mixup_data(x_a, x_b, alpha=5, beta=5):
    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1 
    mixed_x = lam * x_a + (1 - lam) * x_b
    return mixed_x, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)