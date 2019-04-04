from DSLR.utils import load_csv, histogram
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Displaying histogram homogeneous score in the dataset')
    parser.add_argument('-img', help='save plot as a .png image', default=False, action="store_true")
    arguments = parser.parse_args()
    args = dict()
    dataset = load_csv('./datasets/dataset_train.csv')

    args['-img'] = arguments.img
    args['data'] = dataset[1:, 1:]
    args['legend'] = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    args['xlabel'] = 'Marks'
    args['ylabel'] = 'Number of student'
    args['title'] = dataset[0, 16]
    args['color'] = ['red', 'yellow', 'blue', 'green']
    histogram(args)
