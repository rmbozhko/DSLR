import pandas as pd
import toolkit as tl
import argparse
import numpy as np

def     main(args):
    data = pd.read_csv(args.dataset)
    metrics = pd.read_csv(args.weights) 
    faculties = pd.read_csv('./datasets/faculties.csv')

    thetas = metrics.values[:-2, :]
    data = data.fillna(method="ffill")
    X = np.array(data.values[:, [8, 9, 10, 11, 17]], dtype=np.float64)
    sc = tl.StandardScaler(metrics.values[-1, 1:], metrics.values[-2, 1:])
    X_norm = sc.transform(X)
    temp = tl.predict(X_norm, thetas).tolist()
    print("Gryffindor:{}".format(temp.count(2)))
    print("Hufflepuff:{}".format(temp.count(3)))
    print("Ravenclaw:{}".format(temp.count(0)))
    print("Slytherin:{}".format(temp.count(1)))
    with open("houses.csv", 'w') as f:
        print("Index,Hogwarts House", file=f)
        for i in range(len(temp)):
            print("{},{}".format(i, faculties.columns.values[temp[i]]), file=f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
	parser.add_argument('dataset', help='dataset with features to process', type=str)
	parser.add_argument('weights', help='thetas and metrics used to process data and make predictions', type=str)
	args = parser.parse_args()
	main(args)
