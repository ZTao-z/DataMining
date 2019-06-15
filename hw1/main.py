#-*- coding:utf-8 -*-

import csv, os, random, math
import numpy as np
from numpy import *

np.seterr(divide='ignore',invalid='ignore')
np.random.seed(0)

originSet = []
trainSet = []
testSet = []
labelSet = []
theta = mat(np.random.uniform(-4*math.sqrt(6/33),4*math.sqrt(6/33),33) * 0.01)
learningRate = 0.5
lambd = 0.005
lossRate = 1
MAX = None
MIN = None

AVER = None
STANDARD = None

V_dw = 1
V_db = 1

def readTrainingData():
	global originSet
	with open("./data/trainSet.csv", "r") as f:
		f_csv = csv.reader(f)
		headers = next(f_csv)
		for row in f_csv:
			formatRow = [float(n) for n in row]
			originSet.append(formatRow)
	originSet = normalization_maxmin(originSet).tolist()
	
def readTestingData():
	with open("./data/testSet.csv", "r") as f:
		f_csv = csv.reader(f)
		headers = next(f_csv)
		for row in f_csv:
			formatRow = [float(n) for n in row]
			testSet.append(formatRow)

def suffleTrainingSet():
	global originSet, trainSet
	random.shuffle(originSet)

def formatTrainingSet():
	global trainSet, labelSet, originSet
	trainSet = []
	labelSet = []
	for data in originSet:
		trainSet.append(data[0:len(data)-1])
		labelSet.append(1 if int(data[-1]) > 0 else 0)

def formatTestSet(dataSet):
	global MAX, MIN
	#global AVER, STANDARD
	m = len(dataSet)
	matrix = mat(dataSet)
	matrix = (matrix - MIN[0,0:32]) / (MAX[0,0:32] - MIN[0,0:32])
	#matrix = (matrix - AVER) / STANDARD
	matrix = np.insert(matrix, 0, values=[1.0] * m, axis=1)
	return matrix

def CrossValidation(mode):
	global trainSet
	T1 = []
	L1 = []
	T2 = []
	L2 = []
	T3 = []
	L3 = []
	T4 = []
	L4 = []
	T5 = []
	L5 = []
	count = 0
	for i in range(len(trainSet)):
		count += 1
		if count == 1:
			T1.append(trainSet[i])
			L1.append(labelSet[i])
		elif count == 2:
			T2.append(trainSet[i])
			L2.append(labelSet[i])
		elif count == 3:
			T3.append(trainSet[i])
			L3.append(labelSet[i])
		elif count == 4:
			T4.append(trainSet[i])
			L4.append(labelSet[i])
		else:
			T5.append(trainSet[i])
			L5.append(labelSet[i])
			count = 0
	if mode == 0:
		return { "train": mat(T2 + T3 + T4 + T5), "test": mat(T1), "label_tr" : L2 + L3 + L4 + L5, "label_te": L1}
	elif mode == 1:
		return { "train": mat(T3 + T4 + T5 + T1), "test": mat(T2), "label_tr" : L3 + L4 + L5 + L1, "label_te": L2}
	elif mode == 2:
		return { "train": mat(T4 + T5 + T1 + T2), "test": mat(T3), "label_tr" : L4 + L5 + L1 + L2, "label_te": L3}
	elif mode == 3:
		return { "train": mat(T5 + T1 + T2 + T3), "test": mat(T4), "label_tr" : L5 + L1 + L2 + L3, "label_te": L4}
	elif mode == 4:
		return { "train": mat(T1 + T2 + T3 + T4), "test": mat(T5), "label_tr" : L1 + L2 + L3 + L4, "label_te": L5}

def normalization(dataSet):
	global STANDARD, AVER, testSet
	m = len(dataSet)
	matrix = mat(dataSet)
	
	temp_matrix = mat(testSet)
	temp_matrix = np.insert(temp_matrix, 32, values=[0.0] * len(testSet), axis=1)
	allSet = np.insert(matrix, 0, values=temp_matrix, axis=0)
	
	AVER = allSet.mean(axis=0)
	STANDARD = allSet.std(axis=0)
	AVER[0,32] = 0
	STANDARD[0,32] = 1
	#print(AVER, STANDARD)
	matrix = (matrix - AVER) / STANDARD
	matrix = np.insert(matrix, 0, values=[1.0] * m, axis=1)
	return matrix

def normalization_maxmin(dataSet):
	global testSet, MAX, MIN
	m = len(dataSet)
	matrix = mat(dataSet)
	temp_matrix = mat(testSet)
	temp_matrix = np.insert(temp_matrix, 32, values=[0.0] * len(testSet), axis=1)
	allSet = np.insert(matrix, 0, values=temp_matrix, axis=0)
	MAX = allSet.max(axis=0)
	MIN = allSet.min(axis=0)
	matrix = (matrix - MIN) / (MAX - MIN)
	matrix = np.insert(matrix, 0, values=[1.0] * m, axis=1)

	return matrix
	
# dataSet: matrix, label: list
def gradientDescent(dataSet, label):
	global theta
	theta_t = theta
	m,n = theta.shape
	for i in range(n):
		theta_t[0,i] = theta[0, i] - learningRate * J(dataSet, label, i) - learningRate * lambd * theta[0, i] / float(len(dataSet))
	theta = theta_t

def dropout():
	global theta
	m,n = theta.shape
	for i in range(n):
		if abs(theta[0, i]) < 0.0001:
			theta[0,i] = 0

def H(data, theta):
	sum = theta * data.T
	return sum[0,0]

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))

def sigmoid_m(x):
	return 1.0 / (1.0 + np.exp(-x))

def costFunction(dataSet, label):
	global theta, learningRate, lossRate
	m, n = dataSet.shape
	y = mat(label)
	fx = sigmoid_m(theta * dataSet.T)
	sum = -(y * np.log(fx).T + (1 - y) * np.log(1 - fx).T)
	newLoss = (sum / float(m))[0,0]
	if newLoss > lossRate:
		learningRate *= 0.33
		lossRate = newLoss
	print("cost:", newLoss)

# dataSet: matrix, label: list
def J(dataSet, label, num):
	global theta
	sum = 0
	m, n = dataSet.shape
	y = mat(label)
	res = (sigmoid_m(theta * dataSet.T) - y) * dataSet[:, num]
	sum = res[0,0]
	return (float(sum)) / float(m)

def testSingle(data, label):
	label_t = 0
	sum = theta * data.T
	if sigmoid(sum[0,0]) >= 0.5:
		label_t = 1
	return label_t == label

def test(dataSet, label):
	correct = 0
	for i in range(len(dataSet)):
		if testSingle(dataSet[i,:], label[i]):
			correct += 1
	print("correct:",correct / len(dataSet))

def writeResult():
	global theta, testSet
	data = formatTestSet(testSet)
	print(data)
	sum = sigmoid_m(theta * data.T)
	sum = sum.T
	sum = sum > 0.5
	nums = np.arange(1, len(sum)+1).reshape(-1, 1)
	res = np.hstack((nums, sum))
	res = np.vstack((np.array(["id", "Predicted"]), res))
	np.savetxt("res.csv", res, fmt="%s", delimiter=",")

if __name__ == "__main__":
	readTestingData()
	readTrainingData()
	print("finish loading")
	c = 0
	
	while(c < 5):
		print("Iteration : ", c)
		if c % 5 == 0:
			suffleTrainingSet()
			formatTrainingSet()
		stru = CrossValidation(c % 5)
		print("finish cross validation")
		count = 0
		tr = stru["train"]
		te = stru["test"]
		l_tr = stru["label_tr"]
		l_te = stru["label_te"]
		while(count < len(stru["train"])):
			begin = count
			end = len(stru["train"]) if count+256 > len(stru["train"]) else count+256
			t_tr = tr[begin:end]
			t_l_tr = l_tr[begin:end]
			gradientDescent(tr, l_tr)
			count += 256
			test(te, l_te)
			costFunction(te, l_te)
		c += 1
		writeResult()