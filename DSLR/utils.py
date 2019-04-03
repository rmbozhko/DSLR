import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable 
from DSLR.math import calcMean, calcStdDev, calcMin, calcPercentile, calcMax

def		load_csv(filename):
    """
        Reading passed csv file and inserting nan instead of absence
    """
    
    dataset = list()
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile)
        try:
            for row in data:
                line = list()
                for value in row:
                    try:
                        value = float(value)
                    except:
                        if not value:
                            value = np.nan
                    line.append(value)
                dataset.append(line)
        except Exception as exc:
            print("Caught Exception in load_csv: {}".format(str(exc)))
    return np.array(dataset, dtype=object)

def     describe(filename):
    """
        Recreating of np.describe for DSLR School 42 Project
    """
    
    dataset = load_csv(filename)
    features = dataset[0]
    data = dataset[1:, 1:] # omitting index feature
    t = PrettyTable()#['Feature #' + str(i + 1) for i in range(features.shape[0])])
    t.add_column("", ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    # we substract one below, because Index column is not needed to be described
    print(dataset)
    for i in range(features.shape[0] - 1):
        column = np.array(data[:, i], dtype=float)
        if np.all(np.isnan(column)):
            continue
        t.add_column('Feature #' + str(i + 1), ['', calcMean(column), calcStdDev(column), calcMin(column), calcPercentile(column, 25), calcPercentile(column, 50), calcPercentile(column, 75), calcMax(column)])
    print(t)

def     histogram(args):
    """
        Histogram function used to display histogram chart with homogeneous score distribution
    """
    
    for i in range(len(args['legend'])):
        temp = np.array(args['data'][args['data'][:, 1] == args['legend'][i]][:, 16], dtype=np.float64)
        temp = temp[~np.isnan(temp)]
        plt.hist(temp, color=args['color'][i], alpha=0.5)
    
    plt.legend(args['legend'], loc='upper right')
    plt.title(args['title'])
    plt.xlabel(args['xlabel'])
    plt.ylabel(args['ylabel'])
    plt.show()
    plt.savefig('foo.png')
    

