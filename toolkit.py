import numpy as np
from DSLR.math import calcMean, calcStdDev

def             next_batch(X, y, batchSize):
    # loop over our dataset 'X' in mini-batches if size 'batchSize'
    for i in np.arange(0, X.shape[0], batchSize):
        # yield batches of data and labels
        yield X[i:i + batchSize], y[i:i + batchSize]

def             next_mini_batch(X, y, batchSize, shuffle=False):
    assert X.shape[0] == y.shape[0]
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
    for i in range(0, X.shape[0] - batchSize + 1, batchSize):
        if shuffle:
            tiles = indices[i:i + batchSize]
        else:
            tiles = slice(i, i + batchSize)
        yield X[tiles], y[tiles]

def             MBGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=150, sorted=False, lambda_val=0.1, thetas=None, batchSize=16):
    
    # in case, when we have only one feature in X, we can assign m to X.size,
    # otherwise we should specify the axis of X which we are going to assign
    m = Y.shape[0]
    costs = list()

    # Check if sorted, if so shuffle randomly whole dataset
    # Handled both sorted datasets in ASC and DESC order
    for i in range(1, X.shape[1]):
            sorted = all(np.diff(X[:, i]) >= 0) or all(np.diff(X[:, i]) <= 0)
            if sorted:
                    break
    if sorted:
            print("Sorted")
            np.random.shuffle(X)
    
    for j in range(iterationsNum):
        for X, Y in next_mini_batch(X, Y, batchSize):
            g = (h_function(X, thetas.T) - Y).T.dot(X)
            reg = (lambda_val / m) * thetas[:, 1:]
            thetas = thetas - (learningRate * (1 / m) * g + np.insert(reg, 0, 0, axis=1))
            
        # Metrics collecting
        costs.append(computeCost(X, Y, thetas, h_function, lambda_val))
        
    return (thetas, costs)

def             SGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=150, sorted=False, lambda_val=0.1, thetas=None, batchSize=16):

    # in case, when we have only one feature in X, we can assign m to X.size,
    # otherwise we should specify the axis of X which we are going to assign
    m = Y.shape[0]
    costs = list()

    # Check if sorted, if so shuffle randomly whole dataset
    # Handled both sorted datasets in ASC and DESC order
    for i in range(1, X.shape[1]):
            sorted = all(np.diff(X[:, i]) >= 0) or all(np.diff(X[:, i]) <= 0)
            if sorted:
                    break
    if sorted:
            print("Sorted")
            np.random.shuffle(X)

    for j in range(iterationsNum):
        for X, Y in next_batch(X, Y, batchSize):
            g = (h_function(X, thetas.T) - Y).T.dot(X)
            reg = (lambda_val / m) * thetas[:, 1:]
            thetas = thetas - (learningRate * (1 / m) * g + np.insert(reg, 0, 0, axis=1))
            
        # Metrics collecting
        costs.append(computeCost(X, Y, thetas, h_function, lambda_val))
    
    return (thetas, costs)

def		BGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=150, lambda_val=0.1, thetas=None, sorted=False, batchSize=16):
    
    # Setting processing vars
    costs = list()
    m = Y.shape[0] 

    for i in range(iterationsNum):
        g = (h_function(X, thetas.T) - Y).T.dot(X)
        reg = (lambda_val / m) * thetas[:, 1:]
        thetas = thetas - (learningRate * (1 / m) * g + np.insert(reg, 0, 0, axis=1))
 
        # Metrics collecting
        costs.append(computeCost(X, Y, thetas, h_function, lambda_val))
        
    return (thetas, costs)

def train_test_split(X, y, test_size=0.3, random_state=None):
    """Split arrays or matrices into random train and test subsets

    Parameters
    ----------
    X : array-like, shape [n_samples, n_features]
    y: array-like, shape = [n_samples]
    test_size: float, default: 0.3
    random_state: int, default: 0
    """
    if random_state:
        np.random.seed(random_state)

    p = np.random.permutation(len(X))

    X_offset = int(len(X) * test_size)
    y_offset = int(len(y) * test_size)

    X_train = X[p][X_offset:]
    X_test = X[p][:X_offset]

    y_train = y[p][y_offset:]
    y_test = y[p][:y_offset]

    return (X_train, X_test, y_train, y_test)

def		addBiasUnit(arr):
    """
        Adding bias unit to data
    """
    bias_arr = np.ones((arr.shape[0], 1), dtype=np.float64)
    return (np.column_stack((bias_arr, arr)))

def             predict(X, thetas):
    """
        Calculating and retriving the corresponding labels
    """
    X = np.array(addBiasUnit(X))
    predictions = np.array(X.dot(np.array(thetas.T)))
    return (np.argmax(predictions, 1))

def		computeThetas(X, y, gradDesc, h_func, computeCost, faculties):
    """
        Starting point of thetas training
    """
    # setting hyperparameters
    learningRate = 0.01
    lambda_val = 10
    iterationsNum = 50
    batchSize = 16

    # setting training variables
    X = addBiasUnit(X)
    thetas = np.zeros((len(faculties), X.shape[1]), dtype=np.float64)

    # preparing class labels
    y = np.tile(y.reshape((y.shape[0], 1)), 4)
    for k, v in faculties.items():
        mask = y[:, v] == k
        y[:, v] = mask.astype(int)
    y = y.astype(np.float)
    
    [thetas, costs] = gradDesc(X, y, computeCost, h_func, learningRate=learningRate, lambda_val=lambda_val, thetas=thetas, iterationsNum=iterationsNum, batchSize=batchSize)
    return ([thetas, costs])

