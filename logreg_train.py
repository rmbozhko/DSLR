import numpy as np
import argparse
import toolkit as tl
import sys
import math

def computeCostLogReg(X, Y, h_function):
	g = h_function(X)
	J = Y.dot(math.log(g)) + (np.full((Y.shape[0], ), 1) - Y).dot(math.log(np.full((g.shape[0], ), 1) - g))
	return (J / X.shape[0])

def h_function(X):
	return (1 / (1 + sys.float_info.epsilon ** X.dot(tl.thetas)))

def 	predictOneVsAll(X, thetasStorage):
	X = tl.addBiasUnit(X)
	temp = X.dot(thetasStorage.T)

	# selecting max value in range of predicted
	return (np.amax(temp, axis=1))

def 	getOutput(dataset):
	Y = list()

	for x in xrange(0,dataset.shape[0]):
		houseName = dataset[x]['HogwartsHouse'].lower()
		if houseName == 'ravenclaw':
			houseIndex = 0
		elif houseName == 'slytherin':
			houseIndex = 1
		elif houseName == 'gryffindor':
			houseIndex = 2
		elif houseName == 'hufflepuff':
			houseIndex = 3
		else:
			exit("No house found for pupil #" + x)
		Y.append(houseIndex)
	return (np.array(Y))

def 	FindAndFixMissingData(data):

	for x in xrange(0,data.shape[0]):
		if np.isnan(data[x]['Index']):
			data[x]['Index'] = 0
		if not (data[x]['HogwartsHouse'].lower() in ['ravenclaw', 'slytherin', 'gryffindor', 'hufflepuff']):
			data[x]['HogwartsHouse'] = 'Missing'
		if len(data[x]['FirstName']) == 0:
			data[x]['FirstName'] = 'Missing'
		if len(data[x]['LastName']) == 0:
			data[x]['LastName'] = 'Missing'
		if len(data[x]['Birthday']) == 0:
			data[x]['Birthday'] = 'Missing'
		if not (data[x]['BestHand'].lower() in ['left', 'right']):
			data[x]['BestHand'] = 'Missing'	
		if np.isnan(data[x]['Arithmancy']):
			data[x]['Arithmancy'] = np.nan_to_num(data[x]['Arithmancy'])
		if np.isnan(data[x]['Astronomy']):
			data[x]['Astronomy'] = np.nan_to_num(data[x]['Astronomy'])
		if np.isnan(data[x]['Herbology']):
			data[x]['Herbology'] = np.nan_to_num(data[x]['Herbology'])
		if np.isnan(data[x]['DefenseAgainstTheDarkArts']):
			data[x]['DefenseAgainstTheDarkArts'] = np.nan_to_num(data[x]['DefenseAgainstTheDarkArts'])
		if np.isnan(data[x]['Divination']):
			data[x]['Divination'] = np.nan_to_num(data[x]['Divination'])
		if np.isnan(data[x]['MuggleStudies']):
			data[x]['MuggleStudies'] = np.nan_to_num(data[x]['MuggleStudies'])
		if np.isnan(data[x]['AncientRunes']):
			data[x]['AncientRunes'] = np.nan_to_num(data[x]['AncientRunes'])
		if np.isnan(data[x]['HistoryOfMagic']):
			data[x]['HistoryOfMagic'] = np.nan_to_num(data[x]['HistoryOfMagic'])
		if np.isnan(data[x]['Transfiguration']):
			data[x]['Transfiguration'] = np.nan_to_num(data[x]['Transfiguration'])
		if np.isnan(data[x]['Potions']):
			data[x]['Potions'] = np.nan_to_num(data[x]['Potions'])
		if np.isnan(data[x]['CareOfMagicalCreatures']):
			data[x]['CareOfMagicalCreatures'] = np.nan_to_num(data[x]['CareOfMagicalCreatures'])
		if np.isnan(data[x]['Charms']):
			data[x]['Charms'] = np.nan_to_num(data[x]['Charms'])
		if np.isnan(data[x]['Flying']):
			data[x]['Flying'] = np.nan_to_num(data[x]['Flying'])
	return (data)

def 	getFeatures(dataset):
	# function var
	res = np.empty((dataset.shape[0], 13), dtype=float)

	# fullfilling array with data entries
	for x in xrange(0, dataset.shape[0]):
		datapoint = np.array([dataset[x]['Arithmancy'], dataset[x]['Astronomy'], dataset[x]['Herbology'],
							dataset[x]['DefenseAgainstTheDarkArts'], dataset[x]['Divination'], dataset[x]['MuggleStudies'],
							dataset[x]['AncientRunes'], dataset[x]['HistoryOfMagic'], dataset[x]['Transfiguration'],
							dataset[x]['Potions'], dataset[x]['CareOfMagicalCreatures'], dataset[x]['Charms'],
							dataset[x]['Flying']], np.float64)
		res[x] = datapoint

	return (res)

def 	readData(dataset):
	# reading data from a file, except header line
	data = np.genfromtxt(dataset, delimiter=',', skip_header=1, 
		dtype={'names': ('Index', 'HogwartsHouse', 'FirstName', 'LastName', 'Birthday', 'BestHand', 'Arithmancy',
						'Astronomy', 'Herbology', 'DefenseAgainstTheDarkArts', 'Divination', 'MuggleStudies',
						'AncientRunes', 'HistoryOfMagic', 'Transfiguration', 'Potions', 'CareOfMagicalCreatures',
						'Charms', 'Flying'), 
				'formats': ('u2', 'U10', 'U255', 'U255', 'U200', 'U5', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
							'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')})

	data = FindAndFixMissingData(data)
	X = getFeatures(data)
	Y = getOutput(data)
	
	return (X, Y)

def     main(dataset):
	faculties = {'ravenclaw' : 0,  'slytherin' : 1,  'gryffindor' : 2,  'hufflepuff' : 3}
	X, Y = readData(dataset)
	tl.thetas = np.zeros(X.shape[1] + 1)

	if args.is_norm:
		tl.normalEquation(X, Y)
	else:
		if args.is_fscale:
			X = tl.featureScaling(X)
		else:
			X = tl.meanNormalization(X)

		thetasStorage = np.empty((len(faculties), tl.thetas.shape[0]), dtype=float)
		for k, v in faculties.items():
			y = np.equal(Y, np.full((Y.shape[0], ), faculties[k])).astype(int)
			if args.is_sgd:
				[history, iterations] = tl.computeThetas(X, y, tl.SGD, h_function, tl.computeCostSGD)
			else:
				[history, iterations] = tl.computeThetas(X, y, tl.BGD, h_function, tl.computeCostBGD)
			thetasStorage[v] = tl.thetas
	y_pred = predictOneVsAll(X, thetasStorage)
	#print('\nTraining Set Accuracy: {}%'.format(mean(double(y_pred == Y)) * 100));

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further prediction.')
	parser.add_argument('-norm', dest='is_norm', action='store_true', default=False, help='choose normal equation as thetas training algorithm')
	parser.add_argument('-bgd', dest='is_bgd', action='store_true', default=True, help=' [default] choose batch gradient descent as thetas training algorithm')
	parser.add_argument('-sgd', dest='is_sgd', action='store_true', default=False, help='choose stohastic gradient descent as thetas training algorithm')
	parser.add_argument('-meanNorm', dest='is_fscale', action='store_false', default=True, help='choose mean normalization as method to rescale input data')
	parser.add_argument('-fscale', dest='is_fscale', action='store_true', default=True, help=' [default] choose feature scalling as method to rescale input data')
	requiredArgs = parser.add_argument_group('Required arguments')
	requiredArgs.add_argument('-data', help='dataset with input values to train thetas', required=True)
	args = parser.parse_args()
	main(args.data)