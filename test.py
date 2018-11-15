from logreg_train import *
from toolkit import *
import numpy as np

def 	main():
	data = np.genfromtxt("ex2data1.txt", delimiter=',')
	Y = data[:, -1]
	X = data[:, :-1]
	print(X)
	X_old = data[:, :-1]
	
	thetas = np.zeros(X.shape[1] + 1)
	[cost, grad] = computeCostLogReg(X, Y, h_function)	
	print("cost:{} | grad:{}".format(cost, grad))

	thetas = np.array([-24, 0.2, 0.2])
	[cost, grad] = computeCostLogReg(X, Y, h_function)
	print("cost:{} | grad:{}".format(cost, grad))

	#BGD(X, Y, computeCostBGD, h_function)

if __name__ == '__main__':
	main()