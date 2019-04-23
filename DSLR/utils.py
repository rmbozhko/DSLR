import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable 
from DSLR.math import calcMean, calcStdDev, calcMin, calcPercentile, calcMax, calcVar

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
            exit(1)
    return np.array(dataset, dtype=object)

def     describe(filename):
    """
        Recreating of np.describe for DSLR School 42 Project
    """
    
    dataset = load_csv(filename)
    features = dataset[0]
    data = dataset[1:, :]
    t = PrettyTable()
    t.add_column("", ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Var'])
    for i in range(features.shape[0] - 1):
        try:
            column = np.array(data[:, i], dtype=float)
        except Exception:
            continue
        else:
            if np.all(np.isnan(column)):
                continue
        t.add_column(features[i], [data.shape[0], calcMean(column), calcStdDev(column), calcMin(column), calcPercentile(column, 25), calcPercentile(column, 50), calcPercentile(column, 75), calcMax(column), calcVar(column)])
    print(t)

def     histogram(args, ax=None, col=15):
    """
        Histogram function used to display histogram chart with homogeneous score distribution
    """
    for i in range(len(args['legend'])):
        
        temp = np.array(args['data'][(args['data'][:, 0] if not ax else args['faculties']) == args['legend'][i]][:, col], dtype=np.float64)
        temp = temp[~np.isnan(temp)]
        
        if not ax:
            plt.hist(temp, color=args['color'][i], alpha=0.5)
        else:
            ax.hist(temp, color=args['color'][i], alpha=0.5)
    
    if not ax:
        plt.legend(args['legend'], loc='upper right')
        plt.title(args['title'])
        plt.xlabel(args['xlabel'])
        plt.ylabel(args['ylabel'])
        if args['-img']:
            plt.savefig('histogram.png')
        plt.show() 

def     scatter_plot(args, ax=None, xcol=6, ycol=8):
    """
        Scatter_plot function used to display histogram chart with homogeneous score distribution
    """
    for i in range(len(args['legend'])):
        x = np.array(args['data'][(args['data'][:, 0] if not ax else args['faculties']) == args['legend'][i]][:, xcol], dtype=np.float64)
        y = np.array(args['data'][(args['data'][:, 0] if not ax else args['faculties']) == args['legend'][i]][:, ycol], dtype=np.float64)
            
        if not ax:
            plt.scatter(x, y, color=args['color'][i], alpha=0.5)
        else:
            ax.scatter(x, y, color=args['color'][i], s=args['s'], alpha=0.5)
    
    if not ax:
        plt.legend(args['legend'], loc='upper right')
        plt.title(args['title']) # is it correct title?
        plt.xlabel(args['xlabel'])
        plt.ylabel(args['ylabel'])
        if args['-img']:
            plt.savefig('scatter_plot.png')
        plt.show()

def     pair_plot(args):
    """
        Pair_plot function used to display as well as histogram as well as scatter charts concerning features be used in log_reg training
    """
    args['faculties'] = np.array(args['data'][:, 0], dtype=str)
    args['data'] = args['data'][:, 5:]
    size = args['data'].shape[1]
     
    matplotlib.rc('font', **args['font'])
    _, ax = plt.subplots(nrows=size, ncols=size)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    for row in range(size):
        for col in range(size):
            
            if col == row:
                histogram(args, ax[row, col], col)
            else:
                scatter_plot(args, ax[row, col], col, row)
            
            if ax[row, col].is_last_row():
                ax[row, col].set_xlabel(args['features'][col].replace(' ', '\n'))
            else:
                ax[row, col].tick_params(labelbottom=False)
            
            if ax[row, col].is_first_col():
                ax[row, col].set_ylabel(args['features'][row].replace(' ', '\n'))
            else:
                ax[row, col].tick_params(labelleft=False)

    plt.legend(args['legend'], loc='center left', bbox_to_anchor=(1, 0.5))
    if args['-img']:
        plt.savefig('pair_plot.png')
    plt.show()
