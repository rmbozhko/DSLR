import numpy as np
import argparse
from DSLR.utils import describe

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Describing retrieved dataset')
	parser.add_argument('dataset', help='dataset with features to process', type=str)
	args = parser.parse_args()

	describe(args.dataset)
