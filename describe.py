import numpy as np
import sys
import argparse
from logreg_train import readData

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

def calcLastQuirtile(X):
	res = np.empty(X.shape[1], dtype=float)
	pos = X.shape[0] / 2

	# vital part of searching median is sorting beforehand
	X.sort(axis=0)
	for i in xrange(0, X.shape[1]):
		pos = (3.0 / 4.0) * (X.shape[0] + 1)
		if not (pos.is_integer()):
			res[i] = (X[int(pos) - 1][i] + X[int(pos) + 1][i]) / 2 
		else:
			res[i] = X[int(pos)][i]
	return (res)

def calcFirstQuirtile(X):
	res = np.empty(X.shape[1], dtype=float)
	pos = X.shape[0] / 2

	# vital part of searching median is sorting beforehand
	X.sort(axis=0)
	for i in xrange(0, X.shape[1]):
		pos = (1.0 / 4.0) * (X.shape[0] + 1)
		if not (pos.is_integer()):
			res[i] = (X[int(pos) - 1][i] + X[int(pos) + 1][i]) / 2 
		else:
			res[i] = X[int(pos)][i]
	return (res)


def describe(X):
	# function vars
	datasetMean = np.divide(np.sum(X, axis=0), X.shape[0])
	datasetMin = calcMin(X)
	datasetMax = calcMax(X)
	datasetStd = calcStd(X, datasetMean)
	datasetMedian = calcMedian(X)
	datasetLastQuirtile = calcLastQuirtile(X)
	datasetFirstQuirtile = calcFirstQuirtile(X)
	
	#print(np.percentile(X, 75, axis=0))
	#print(datasetLastQuirtile)
	#print(np.percentile(X, 25, axis=0))
	#print(datasetFirstQuirtile)
	
	'''sys.stdout.write("Feature ")

	for x in xrange(0, X.shape[1]):
		sys.stdout.write("{}\t".format(x + 1))	
	
	sys.stdout.write("\nCount\t")
	
	for x in xrange(0, X.shape[1]):
		sys.stdout.write("{}\t".format(X.shape[0]))
	
	sys.stdout.write("\nMean\t")

	for x in xrange(0, X.shape[1]):
		sys.stdout.write("{}\t".format(datasetMean[x]))'''


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