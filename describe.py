import numpy as np
import argparse
from logreg_train import readData
from prettytable import PrettyTable

def 	calcStd(X, mean):
	res = np.empty(X.shape[1], dtype=float)
	
	for i in xrange(0, X.shape[1]):
		stdStorage = np.empty(X.shape[0], dtype=float)
		for j in xrange(0, X.shape[0]):
			stdStorage[j] = np.square(X[j][i] - mean[i])
		tempMean = np.divide(np.sum(stdStorage, axis=0), X.shape[0])
		res[i] = np.sqrt(tempMean)
	return (res)

def 	calcMax(X):
	res = np.empty(X.shape[1], dtype=float)
	
	for i in xrange(0, X.shape[1]):
		summation = X[0][i]
		for j in xrange(0, X.shape[0]):
			if X[j][i] > summation:
				summation = X[j][i]
		res[i] = summation
	return (res)

def 	calcMin(X):
	res = np.empty(X.shape[1], dtype=float)
	
	for i in xrange(0, X.shape[1]):
		summation = X[0][i]
		for j in xrange(0, X.shape[0]):
			if X[j][i] < summation:
				summation = X[j][i]
		res[i] = summation
	return (res)

def 	calcMedian(X):
	# function vars
	res = np.empty(X.shape[1], dtype=float)
	pos = X.shape[0] / 2

	# vital part of searching median is sorting beforehand
	X.sort(axis=0)

	for i in xrange(0, X.shape[1]):
		if (X.shape[0] % 2) == 0:
			res[i] = (X[pos][i] + X[pos - 1][i]) / 2
		else:
			res[i] = X[pos][i]
	return (res)

def calcPercentile(X, quirtile):
	# function vars
	res = np.empty(X.shape[1], dtype=float)
	is_integer = True
	pos = quirtile * (X.shape[0] + 1)

	# vital part of searching median is sorting beforehand
	X.sort(axis=0)

	if not (pos.is_integer()):
		pos = int(pos)
		is_integer = False

	for i in xrange(0, X.shape[1]):
		if not is_integer:
			res[i] = (X[pos - 1][i] + X[pos + 1][i]) / 2
		else:
			res[i] = X[pos][i]
	return (res)

def tabulateData(metrics, colName):
	temp = list()
	temp.append(colName)

	for x in xrange(0, metrics.shape[0]):
		temp.append(str(metrics[x]))
	
	return (temp)

def describe(X):
	# calculating metrics
	datasetMean = np.divide(np.sum(X, axis=0), X.shape[0])
	datasetMin = calcMin(X)
	datasetMax = calcMax(X)
	datasetStd = calcStd(X, datasetMean)
	datasetMedian = calcMedian(X)
	datasetLastQuirtile = calcPercentile(X, .75)
	datasetFirstQuirtile = calcPercentile(X, .25)

	# displaying metrics
	t = PrettyTable(['Feature ' + str(x) if x is not 0 else '' for x in xrange(0, X.shape[1] + 1)])
	
	t.add_row(tabulateData(np.full((X.shape[1], ), X.shape[0], dtype=int), 'Count'))
	t.add_row(tabulateData(datasetMean, 'Mean'))
	t.add_row(tabulateData(datasetStd, 'Std'))
	t.add_row(tabulateData(datasetMin, 'Min'))
	t.add_row(tabulateData(datasetFirstQuirtile, '25%'))
	t.add_row(tabulateData(datasetMedian, '50%'))
	t.add_row(tabulateData(datasetLastQuirtile, '75%'))
	t.add_row(tabulateData(datasetMax, 'Max'))
	print(t)
	

def     main(dataset):
	global thetas
	
	X, Y = readData(dataset)
	describe(X)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Describing retrieved dataset')
	requiredArgs = parser.add_argument_group('Required arguments')
	requiredArgs.add_argument('-data', help='dataset with features to process', required=True)
	args = parser.parse_args()
	main(args.data)