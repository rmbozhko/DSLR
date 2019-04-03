from DSLR.utils import load_csv, histogram

if __name__ == '__main__':
    args = dict()
    dataset = load_csv('./datasets/dataset_train.csv')
    
    args['data'] = dataset[1:, :]
    args['legend'] = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    args['xlabel'] = 'Marks'
    args['ylabel'] = 'Number of student'
    args['title'] = dataset[0, 16]
    args['color'] = ['red', 'yellow', 'blue', 'green']
    histogram(args)
