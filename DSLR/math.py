import numpy as np

def 	calcVar(X):
    """
    Calculating the variance value of passed array
    """

    mean = calcMean(X)
    res = 0
    for value in X:
        if np.isnan(value):
            continue
        res = res + np.square(value - mean)
    return (np.divide(res, X.shape[0]))


def 	calcStdDev(X):
    """
    Calculating the standard deviation value of passed array
    """

    mean = calcMean(X)
    res = 0
    for value in X:
        if np.isnan(value):
            continue
        res = res + np.square(value - mean)
    return (np.sqrt(np.divide(res, X.shape[0])))

def 	calcMax(X):
    """
    Calculating the max value of passed array
    """
    
    max_val = X[0]
    for value in X:
        if np.isnan(value):
            continue
        if value > max_val:
            max_val = value
    return (max_val)

def 	calcMin(X):
    """
    Calculating the min value of passed array
    """
    
    min_val = X[0]
    for value in X:
        if np.isnan(value):
            continue
        if value < min_val:
            min_val = value
    return (min_val)

def     calcMean(X):
    """
    Calculating the min value of passed array
    """
    
    summ = 0
    for value in X:
        if np.isnan(value):
            continue
        summ = summ + value

    return (summ / X.shape[0])
    

def calcPercentile(X, quirtile):
    pos = (quirtile / 100) * (X.shape[0] + 1)
    ubond = np.ceil(pos)
    lbond = np.floor(pos)
    
    # vital part of searching median is sorting beforehand
    X.sort()
    
    if ubond == lbond:
        return X[int(pos)]
        
    return ((pos - lbond) * (X[int(ubond)] - X[int(lbond)]) + X[int(lbond)])
