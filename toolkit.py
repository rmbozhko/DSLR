import numpy as np
from DSLR.math import calcMean, calcStdDev

thetas = None 
"""
def     meanNormalization(data):
    return (np.divide(data - np.mean(data), np.std(data)))
"""
def 	SGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=150, sorted=False, lambda_val=0.1):
	global thetas

	# in case, when we have only one feature in X, we can assign m to X.size,
	# otherwise we should specify the axis of X which we are going to assign
	m = Y.shape[0]

	# Check if sorted, if so shuffle randomly whole dataset
	# Handled both sorted datasets in ASC and DESC order
	for i in range(1, X.shape[1]):
		sorted = all(np.diff(X[:, i]) >= 0) or all(np.diff(X[:, i]) <= 0)
		if sorted:
			break
	if sorted:
		print("Sorted")
		np.random.shuffle(X)
	
	# Metrics storages
	thetasHistory = list()
	iterations = list()
	
	for j in range(iterationsNum):
		for i in range(X.shape[0]):
			X_temp = np.array([X[i, :]])
			J = (h_function(X_temp) - Y[i]).dot(X_temp)
			J = learningRate * J
			thetas = thetas - J
		# Metrics collecting
		thetasHistory.append(computeCost(X, Y, h_function, lambda_val))
		iterations.append(j)

	return (iterations, thetasHistory)

def		BGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=1500, lambda_val=0.1):
    global thetas
    
    # in case, when we have only one feature in X, we can assign m to X.size,
	# otherwise we should specify the axis of X which we are going to assign
    m = Y.shape[0] 
    temp_thetas = thetas

    # Metrics storages
    thetasHistory = list()
    iterations = list()
    #temp_thetas[0] = 0.0
    print(thetas)
    print(thetas.shape)
    print(thetas[1:])
    print("_____________________")
    for i in range(iterationsNum):
        #J = (h_function(X) - Y).dot(X)
        #J = np.divide(J, m)
        #J = J + ((lambda_val / m) * temp_thetas)
        #thetas = thetas - (learningRate * J)
        
        J = h_function(X)
        reg = (lambda_val / m) * thetas[1:]
        #print(reg.shape)
        #print(np.insert(reg, 0, 0))
        #exit(0)
        thetas = thetas - (learningRate * (1 / m) * (J - Y).dot(X) + np.insert(reg, 0, 0))
        print("Bias Unit: {}".format(thetas[0]))
        # Metrics collecting
        thetasHistory.append(computeCost(X, Y, h_function, lambda_val))
        iterations.append(i)
        
    return (iterations, thetasHistory)

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
	bias_arr = np.ones((arr.shape[0], 1), dtype=np.float64)
	return (np.column_stack((bias_arr, arr)))

def     predict(X, thetas):
    X = np.array(addBiasUnit(X))
    predictions = np.array(X.dot(np.array(thetas.T)))
    temp = np.argmax(predictions, 1)
    return (temp)

def		computeThetas(X, y, gradDesc, h_func, computeCost, lambda_val):
    global thetas
    
    thetas = np.zeros(X.shape[1] + 1, dtype=np.float64)
    learningRate = 0.01
    
    # adding bias column to X data
    X = addBiasUnit(X)
    
    [iterations, history] = gradDesc(X, y, computeCost, h_func, learningRate, lambda_val=lambda_val)
    return ([thetas, history, iterations])

class StandardScaler(object):
  """Standardize features by removing the mean and scaling to unit variance

  Attributes
  ----------
  _mean: 1d-array, shape [n_features]
    Mean of the training samples or zero
  _std: 1d-array, shape [n_features]
    Standard deviation of the training samples or one
  """
  def __init__(self, mean=np.array([]), std=np.array([])):
    self._mean = mean
    self._std = std

  def fit(self, X):
    """Compute the mean and std to be used for later scaling.

    Parameters
    ----------
    X : array-like, shape [n_samples, n_features]
    """
    for i in range(0, X.shape[1]):
      self._mean = np.append(self._mean, calcMean(X[:, i]))
      self._std = np.append(self._std, calcStdDev(X[:, i]))

  def transform(self, X):
    """Perform standardization by centering and scaling

    Parameters
    ----------
    X : array-like, shape [n_samples, n_features]
    """
    return ((X - self._mean) / self._std)
