import numpy as np
import argparse
import toolkit as tl
import sys
import pandas as pd

def     computeCostLogReg(X, Y, h_function, lambda_val):
    temp_thetas = tl.thetas
    m = X.shape[0]
    g = h_function(X)
    temp_thetas[0] = 0.0
    
    J = Y.dot(np.log(g)) + (np.full((Y.shape[0], ), 1) - Y).dot(np.log(np.full((g.shape[0], ), 1.0) - g))
    J = -(J / m)
    J = J + ((lambda_val / 2 * m) * np.sum(np.square(temp_thetas), axis=0)) 
    
    return (J)

def     h_function(X):
    temp = np.array(-1.0 * X.dot(tl.thetas), dtype=np.float64)
    return (1.0 / (1.0 + (np.exp(temp))))

def     main(args):
    data = pd.read_csv(args.dataset)
    data = data.dropna(subset=["Defense Against the Dark Arts", "Charms", "Herbology", "Divination", "Muggle Studies"])
    faculties = {'Ravenclaw' : 0,  'Slytherin' : 1,  'Gryffindor' : 2,  'Hufflepuff' : 3}
    y = np.array(data.values[:, 1], dtype=str) 
    X = np.array(data.values[:, [8, 9, 10, 11, 17]], dtype=np.float64)
    X = tl.meanNormalization(X)
    thetasStorage = np.empty((len(faculties), X.shape[1] + 1), dtype=np.float64)
    lambda_val = 0.1
    for k, v in faculties.items():
        if args.is_sgd:
            [thetas, history, iterations] = tl.computeThetas(X, (y == k).astype(int), tl.SGD, h_function, computeCostLogReg, lambda_val=lambda_val)
        else:
            [thetas, history, iterations] = tl.computeThetas(X, (y == k).astype(int), tl.BGD, h_function, computeCostLogReg, lambda_val=lambda_val)
        thetasStorage[v] = thetas

    plt.plot(iterations, history)
    plt.ylabel('Function cost')
    plt.xlabel('Iterations')
    if args.img:
        plt.savefig('LogRegTraining.png')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
	parser.add_argument('-bgd', dest='is_bgd', action='store_true', default=True, help=' [default] choose batch gradient descent as thetas training algorithm')
	parser.add_argument('-sgd', dest='is_sgd', action='store_true', default=False, help='choose stohastic gradient descent as thetas training algorithm')
	parser.add_argument('-img', dest='is_img', action='store_true', default=False, help='save .png image of the plot')
	parser.add_argument('dataset', help='dataset with features to process', type=str)
	args = parser.parse_args()
	main(args)
