import matplotlib.pyplot as plt
import numpy as np
import argparse
import toolkit as tl
import pandas as pd
from DSLR.math import calcMean, calcStdDev
from sklearn.metrics import accuracy_score

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
    #X = tl.meanNormalization(X)
    X_train, X_test, y_train, y_test = tl.train_test_split(X, y, test_size=0.3, random_state=4)
    sc = tl.StandardScaler()
    sc.fit(X_train)

    X_train_norm = sc.transform(X_train)
    X_test_norm = sc.transform(X_test)
    X = X_train_norm
    y = y_train

    thetasStorage = np.empty((len(faculties), X.shape[1] + 1), dtype=np.float64)
    lambda_val = 10
    for k, v in faculties.items():
        if args.is_sgd:
            [thetas, history, iterations] = tl.computeThetas(X, (y == k).astype(int), tl.SGD, h_function, computeCostLogReg, lambda_val=lambda_val)
        else:
            [thetas, history, iterations] = tl.computeThetas(X, (y == k).astype(int), tl.BGD, h_function, computeCostLogReg, lambda_val=lambda_val)
        thetasStorage[v] = thetas

    y_pred = tl.predict(X_test_norm, thetasStorage).tolist()
    temp = { v:k for k, v in faculties.items() }
    y_pred = [temp.get(item, item) for item in y_test]
    #print("+++++++++++++++++++++")
    #print(y_pred)
    #print(y_test)
    print("Missclasified samples: {}".format(sum(y_test != y_pred)))
    print("Accuracy: %2.f" % accuracy_score(y_test, y_pred))
    save_model(thetasStorage, faculties, sc)
    plt.plot(iterations, history)
    plt.ylabel('Function cost')
    plt.xlabel('Iterations')
    if args.is_img:
        plt.savefig('LogRegTraining.png')

def     save_model(thetasStorage, faculties, sc):
    df = pd.DataFrame(faculties, index=[0])
    df.to_csv('./datasets/faculties.csv', index=False, mode='w+')
    squeezed_metrics = np.column_stack((np.insert(sc._std, 0, 0.0), np.insert(sc._mean, 0, 0.0))).T
    metrics = np.concatenate((thetasStorage, squeezed_metrics), axis=0)
    df = pd.DataFrame(metrics)
    df.to_csv('./datasets/weights.csv', index=False, mode='w+')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
	parser.add_argument('-bgd', dest='is_bgd', action='store_true', default=True, help=' [default] choose batch gradient descent as thetas training algorithm')
	parser.add_argument('-sgd', dest='is_sgd', action='store_true', default=False, help='choose stohastic gradient descent as thetas training algorithm')
	parser.add_argument('-img', dest='is_img', action='store_true', default=False, help='save .png image of the plot')
	parser.add_argument('dataset', help='dataset with features to process', type=str)
	args = parser.parse_args()
	main(args)
