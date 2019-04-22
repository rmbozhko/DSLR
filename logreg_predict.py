import pandas as pd
import toolkit as tl
import argparse
import numpy as np

def     main(args):

    # reading data from sources
    data = pd.read_csv(args.dataset)
    metrics = pd.read_csv(args.weights)
    faculties = pd.read_csv('./datasets/faculties.csv')

    # setting prediction variables
    mean = metrics.values[-1, 1:]
    std = metrics.values[-2, 1:]
    max = metrics.values[-3, 1:]
    thetas = metrics.values[:-3, :]

    # setting and preprocessing test dataset
    data = data.fillna(method="ffill")
    X = np.array(data.values[:, [8, 9, 10, 11, 17]], dtype=np.float64)
    if not args.is_fscale:
        X_norm = (X - mean) / std
    else:
        X_norm = X / max

    # Predicting labels using thetas and writing predicted values to a file
    temp = tl.predict(X_norm, thetas).tolist()
    with open("houses.csv", 'w') as f:
        print("Index,Hogwarts House", file=f)
        for i in range(len(temp)):
            print("{},{}".format(i, faculties.columns.values[temp[i]]), file=f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
	parser.add_argument('dataset', help='dataset with features to process', type=str)
	parser.add_argument('weights', help='thetas and metrics used to process data and make predictions', type=str)
	parser.add_argument('-fscale', dest='is_fscale', action='store_true', default=False, help='switch the feature scaling as features preprocessing functions')
	args = parser.parse_args()
	main(args)
