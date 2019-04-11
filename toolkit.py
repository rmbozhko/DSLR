import numpy as np

thetas = None 

def 	featureScaling(data):
	max_value = np.amax(data, axis=0)
	return (np.divide(data, max_value))

def     meanNormalization(data):
    return (np.divide(data - np.mean(data), np.std(data)))

def 	computeCostSGD(X, Y, h_func):
	J = h_func(X)
	J = np.sum(np.square(J - Y) / 2) / (Y.shape[0])
	return (J)

def 	computeCostBGD(X, Y, h_func):
	J = h_func(X)
	J = np.sum(np.square(J - Y)) / (2 * Y.shape[0])
	return (J)

def 	SGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=150, sorted=False):
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
		thetasHistory.append(computeCost(X, Y))
		iterations.append(j)

	return (iterations, thetasHistory)

def		BGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=1500):
	global thetas

	# in case, when we have only one feature in X, we can assign m to X.size,
	# otherwise we should specify the axis of X which we are going to assign
	m = Y.shape[0]
	
	# Metrics storages
	thetasHistory = list()
	iterations = list()

	for i in range(iterationsNum):
		J = (h_function(X) - Y).dot(X)
		J = learningRate * np.divide(J, m)
		thetas = thetas - J
		
		# Metrics collecting
		thetasHistory.append(computeCost(X, Y, h_function))
		iterations.append(i)

	return (iterations, thetasHistory)

def		addBiasUnit(arr):
	bias_arr = np.ones((arr.shape[0], 1), dtype=np.float64)
	return (np.column_stack((bias_arr, arr)))

def		calcAccuracy(X, Y, logReg=True):
	global thetas

	if logReg:
		temp_y = X.dot(thetas)
		pred = np.mean(Y == temp_y) * 100
	else:
		pred = int(np.sum(Y - X.dot(thetas)))
	return (pred)

def		computeThetas(X, y, gradDesc, h_func, computeCost, lambda_val, num_labels):
    global thetas
    
    thetas = np.zeros(X.shape[1] + 1, dtype=np.float64)
    # adding bias column to X data
    X = addBiasUnit(X)
    # cycle vars
    diff = 1
    prev_diff = 0
    learningRate = 1.0
    step = 0.1
	
	# determing best-fitting learningRate using brute-force	
    while abs(diff) > 0.00000001:
            prev_diff = diff 
            for i in range(9):
                if diff <= 0.0001 and diff >= 0.:
                    break
                learningRate = learningRate - step
                [iterations, history] = gradDesc(X, y, computeCost, h_func, learningRate)
                diff = calcAccuracy(X, y, False)
            step = step * 0.1
            if (diff == prev_diff):
                break
            print('learningRate:{}'.format(learningRate))
    return ([thetas, history, iterations])
