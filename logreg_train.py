import numpy as np
import argparse
import toolkit as tl
import sys
import math
import pandas as pd

def computeCostLogReg(X, Y, h_function):
	g = h_function(X)
	J = Y.dot(math.log(g)) + (np.full((Y.shape[0], ), 1) - Y).dot(math.log(np.full((g.shape[0], ), 1) - g))
	return -(J / X.shape[0])

def     h_function(X):
    temp = np.array(-1.0 * X.dot(tl.thetas), dtype=np.float64)
    return (1.0 / (1.0 + (np.exp(temp))))

def 	predictOneVsAll(X, thetasStorage):
	X = tl.addBiasUnit(X)
	temp = X.dot(thetasStorage.T)
	# selecting max value in range of predicted
	return (np.amax(temp, axis=1))

def     main(dataset):
    data = pd.read_csv(dataset)
    data = data.dropna(subset=["Defense Against the Dark Arts", "Charms", "Herbology", "Divination", "Muggle Studies"])
    print(data.shape)
    faculties = {'Ravenclaw' : 0,  'Slytherin' : 1,  'Gryffindor' : 2,  'Hufflepuff' : 3}
    y = np.array(data.values[:, 1], dtype=str) 
    X = np.array(data.values[:, [8, 9, 10, 11, 17]], dtype=np.float64)
    print(X)
    print(y)
    #X = tl.featureScaling(X)
    X = tl.meanNormalization(X)
    print(X)
    thetasStorage = np.empty((len(faculties), X.shape[1] + 1), dtype=np.float64)
    print(thetasStorage.shape)
    lambda_val = 0.1
    exit(0)
    for k, v in faculties.items():
        print(X)
        print(y)
        if args.is_sgd:
            [thetas, history, iterations] = tl.computeThetas(X, (y == k).astype(int), tl.SGD, h_function, computeCostLogReg, lambda_val=lambda_val, num_labels=len(faculties))
        else:
            [thetas, history, iterations] = tl.computeThetas(X, (y == k).astype(int), tl.BGD, h_function, computeCostLogReg, lambda_val=lambda_val, num_labels=len(faculties))
        thetasStorage[v] = thetas

	#y_pred = predictOneVsAll(X, thetasStorage)
	#print(y_pred)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
	parser.add_argument('-bgd', dest='is_bgd', action='store_true', default=True, help=' [default] choose batch gradient descent as thetas training algorithm')
	parser.add_argument('-sgd', dest='is_sgd', action='store_true', default=False, help='choose stohastic gradient descent as thetas training algorithm')
	parser.add_argument('dataset', help='dataset with features to process', type=str)
	args = parser.parse_args()
	main(args.dataset)
