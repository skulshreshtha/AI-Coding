# Sigmoid function or the logistic function limits the output of any function to (-1,1)
import numpy as np

def sigmoid(x):
    return 1/(1+ np.exp(-x))