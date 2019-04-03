from DSLR.utils import load_csv, scatter_plot
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Displaying histogram homogeneous score in the dataset')
    parser.add_argument('-img', help='save plot as a .png image', default=False, action="store_true")
    arguments = parser.parse_args()
    args = dict()
    dataset = load_csv('./datasets/dataset_train.csv')

    args['-img'] = arguments.img
    args['data'] = dataset[1:, :]
    args['legend'] = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    args['xlabel'] = dataset[0, 7]
    args['ylabel'] = dataset[0, 9]
    args['title'] = 'Scatter plot'
    args['color'] = ['red', 'yellow', 'blue', 'green']
    scatter_plot(args)
