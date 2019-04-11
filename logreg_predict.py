import pandas as pd
import toolkit as tl
import argparse
import numpy as np

def 	predictOneVsAll(X, thetasStorage):
	X = tl.addBiasUnit(X)
	temp = X.dot(thetasStorage.T)
	# selecting max value in range of predicted
	return (np.amax(temp, axis=1))

def     main(args):
    data = pd.read_csv(args.dataset)
    faculties = pd.read_csv('./datasets/faculties.csv')
    thetas = pd.read_csv('./datasets/weights.csv')
    metrics = pd.read_csv('./datasets/metrics.csv')
    data = data.fillna(method="ffill")
    X = np.array(data.values[:, [8, 9, 10, 11, 17]], dtype=np.float64)
    X = np.divide(X - metrics.values[0][0], metrics.values[0][1])
    print(X.shape)
    X = np.array(tl.addBiasUnit(X))
    print(X.shape)
    print(thetas.T.shape)
    print(faculties)
    predictions = np.array(X.dot(np.array(thetas.T)))
    print(predictions)
    print(predictions.shape)
    temp = np.argmax(predictions, 1)
    for i in temp:
        print(faculties.columns.values[i])

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
	parser.add_argument('dataset', help='dataset with features to process', type=str)
	args = parser.parse_args()
	main(args)
