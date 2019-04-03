import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable 
from DSLR.math import calcMean, calcStdDev, calcMin, calcPercentile, calcMax

def		load_csv(filename):
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
                        value = np.nan
                    line.append(value)
                dataset.append(line)
        except Exception as exc:
            print("Caught Exception in load_csv: {}".format(str(exc)))
    return np.array(dataset, dtype=object)

def describe(filename):
    dataset = load_csv(filename)
    features = dataset[0]
    data = dataset[1:, 1:] # omitting index feature
    t = PrettyTable()#['Feature #' + str(i + 1) for i in range(features.shape[0])])
    t.add_column("", ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    # we substract one below, because Index column is not needed to be described
    for i in range(features.shape[0] - 1):
        column = np.array(data[:, i], dtype=float)
        if np.all(np.isnan(column)):
            continue
        t.add_column('Feature #' + str(i + 1), ['', calcMean(column), calcStdDev(column), calcMin(column), calcPercentile(column, 25), calcPercentile(column, 50), calcPercentile(column, 75), calcMax(column)])
    print(t)
	
	

