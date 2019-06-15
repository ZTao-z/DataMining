#coding:utf-8
import random, os, time
from numpy import *
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.joblib import Parallel,delayed
from sklearn.tree import export_graphviz
import pandas as pd

def readDataset():
	trainSet = []
	labelSet = []
	for i in range(1, 6):
		trainData = pd.read_csv(os.path.join('data/', 'train{}.csv'.format(i)), header=None, \
			delimiter="\t", quoting=3)
		labelData = pd.read_csv(os.path.join('data/', 'label{}.csv'.format(i)), header=None, \
			delimiter="\t", quoting=3)
		for example in list(trainData[0]):
			cur_example = example.strip().split(',')
			fin_example = map(float, cur_example)
			trainSet.append(list(fin_example))
		for label in list(labelData[0]):
			labelSet.append(float(label))
	tS_matrix = trainSet
	tL_matrix = labelSet
	
	return tS_matrix, tL_matrix

def readTestSet():
	testset = []
	for i in range(1, 5):
		testData = pd.read_csv(os.path.join('data/', 'test{}.csv'.format(i)), header=None, \
			delimiter="\t", quoting=3)
		for example in list(testData[0]):
			cur_example = example.strip().split(',')
			fin_example = map(float, cur_example)
			testset.append(list(fin_example))

	return testset
	
if __name__=="__main__":
	
	print("read dataset")
	trainData, label = readDataset()
	print(shape(trainData))
	print(shape(label))
	print("begin generate forest")
	begin = time.time()
	clf = RandomForestRegressor(n_estimators = 1, n_jobs=2)
	clf.fit(trainData, label)
	end = time.time()
	print(end-begin)
	
	test = readTestSet()

	result = clf.predict(test)
	
	r = list(range(1,len(test)+1))
	output = pd.DataFrame( data={"id": r, "Predicted":result} )

	# Use pandas to write the comma-separated output file
	output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'result.csv'), index=False, quoting=3)