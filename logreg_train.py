import matplotlib.pyplot as plt
import numpy as np
import argparse
import toolkit as tl
import pandas as pd
from DSLR.math import calcMean, calcMax, calcStdDev
from sklearn.metrics import accuracy_score

def     computeCostLogReg(X, Y, thetas, h_function, lambda_val):
    """
        Computing cost of function with passed thetas
    """
    # setting processing vars
    m = X.shape[0]
    g = h_function(X, thetas.T)
    ones = np.ones((Y.shape[0], 1), dtype=np.float64)
    
    reg_term = (lambda_val / (2 * m)) * np.sum(np.sum(np.square(thetas[:, 1:]), axis=1))
    J = -(1 / m) * sum(Y.T.dot(np.log(g)) + (ones - Y).T.dot(np.log(ones - g))) + reg_term
    return (J)

def     h_function(X, thetas):
    """
        Sigmoid function used with Logistic Regression to calculate the prediction values
    """
    temp = np.array(-1.0 * X.dot(thetas), dtype=np.float64)
    return (1.0 / (1.0 + (np.exp(temp))))

def     ft_get_metrics(X, metricsFunc):
    """
        Calculating the metrics of passed dataset using metricsFunction
    """
    metrics = np.empty([X.shape[1]], dtype=np.float64)
    for col in range(X.shape[1]):
        metrics[col] = metricsFunc(X[:, col])
    return (metrics)

def     main(args):
    hyperparameters = dict()
    data = pd.read_csv(args.dataset)
    data = data.dropna(subset=["Defense Against the Dark Arts", "Charms", "Herbology", "Divination", "Muggle Studies"])
    faculties = {'Ravenclaw' : 0,  'Slytherin' : 1,  'Gryffindor' : 2,  'Hufflepuff' : 3}
    y = np.array(data.values[:, 1], dtype=str) 
    X = np.array(data.values[:, [8, 9, 10, 11, 17]], dtype=np.float64)
    X_train, X_test, y_train, y_test = tl.train_test_split(X, y, test_size=0.3, random_state=4)
    mean = ft_get_metrics(X_train, calcMean)
    max = ft_get_metrics(X_train, calcMax)
    std = ft_get_metrics(X_train, calcStdDev)
    hyperparameters['iterations'] = args.iterations
    hyperparameters['alpha'] = args.alpha if args.alpha > 1e-6 and args.alpha < 1e+2 else 0.01
    hyperparameters['lambda_val'] = args.lambda_val if args.lambda_val > 1e-6 and args.lambda_val < 1e+2 else 10
    hyperparameters['batch_size'] = args.batch_size

    # Normalizing data points
    if not args.is_fscale:
        X_train_norm = ((X_train - mean) / std)
        X_test_norm = ((X_test - mean) / std)
    else:
        X_train_norm = X_train / max
        X_test_norm = X_test / max

    if args.is_sgd:
        [thetas, costs] = tl.computeThetas(X_train_norm, y_train, tl.SGD, h_function, computeCostLogReg, faculties, hyperparameters)
    elif args.is_bgd:
        [thetas, costs] = tl.computeThetas(X_train_norm, y_train, tl.BGD, h_function, computeCostLogReg, faculties, hyperparameters)
    else:
        print("No training algorithm was choosed")
        exit(1)
 
    # Predicting values of the test set and displaying accuracy of trained thetas
    y_pred = tl.predict(X_test_norm, thetas).tolist()
    temp = { v:k for k, v in faculties.items() }
    y_pred = [temp.get(item, item) for item in y_test]    
    print("Missclasified samples: {}".format(sum(y_test != y_pred)))
    print("Accuracy: %2.f" % accuracy_score(y_test, y_pred))
    
    # saving thetas and metrics to files
    save_model(thetas, faculties, [mean, max, std])
    
    # plotting the retrieved metrics
    plotting_data(range(1, len(costs) + 1), costs, 'Iterations', 'Cost Function', 'Logistic Regression -- Cost function improving', args.is_img)
    #plotting_data(range(1, len(errors) + 1), errors, 'Missclassified entries', 'Iterations', 'Logistic Regression -- Improving missclassification examples', args.is_img)

def     plotting_data(X, y, xlabel, ylabel, title, is_img=False):
    plt.plot(X, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()
    if is_img:
        plt.savefig(title + 'png')

def     save_model(thetas, faculties, metrics):
    df = pd.DataFrame(faculties, index=[0])
    df.to_csv('./datasets/faculties.csv', index=False, mode='w+')
    squeezed_metrics = np.column_stack((np.insert(metrics[1], 0, 0.0), np.insert(metrics[2], 0, 0.0)))
    squeezed_metrics = np.column_stack((squeezed_metrics, np.insert(metrics[0], 0, 0.0))).T
    metrics = np.concatenate((thetas, squeezed_metrics), axis=0)
    df = pd.DataFrame(metrics)
    df.to_csv('./datasets/weights.csv', index=False, mode='w+')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
    parser.add_argument('--lambda', dest='lambda_val', type=float, default=10.0, help='lamda values used in regularization term')
    parser.add_argument('--iterations', dest='iterations', type=int, choices=range(1, 2000), default=50, help='Number of iterations we use with gradient descent', metavar="(1, 2000)")
    parser.add_argument('--batchSize', dest='batch_size', type=int, choices=range(1, 64), default=16, help='The size of batches we split dataset to in SGD an Mini-batch gradient descent', metavar="(1, 64)")
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.01, help='Alpha hyperparameter that we use to accelerate moving to function global minimum')
    parser.add_argument('-bgd', dest='is_bgd', action='store_true', default=True, help=' [default] choose batch gradient descent as thetas training algorithm')
    parser.add_argument('-sgd', dest='is_sgd', action='store_true', default=False, help='choose stohastic gradient descent as thetas training algorithm')
    parser.add_argument('-img', dest='is_img', action='store_true', default=False, help='save .png image of the plot')
    parser.add_argument('-fscale', dest='is_fscale', action='store_true', default=False, help='switch the feature scaling as features preprocessing functions')
    parser.add_argument('dataset', help='dataset with features to process', type=str)
    args = parser.parse_args()
    main(args)
