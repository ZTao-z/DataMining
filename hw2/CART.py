import random, os, time
from numpy import *
import numpy as np
import pandas as pd
import multiprocessing, threading

from sklearn.externals.joblib import Parallel, delayed

'''
dataset为matrix，且最后一列为label

算法流程：
# 随机筛选N个样本
# 随机抽取M个特征
# 遍历N个样本上的特征m，找到最优分割阈值，分割成两堆
	记录此特征及阈值
	若满足分裂条件：
		生成此结点的左右子节点，左节点小于该阈值，右节点大于该阈值
		将数据分成两堆，分别保存到左右节点
	否则：
		终止分裂
'''

class CART_tree:
	def __init__(self):
		# data为样本序号
		self.stopNodeSampleNum = 100
		self.varThres = 1
	
	# 划分数据集
	def datasetSplit(self, dataset, feaNum, thres):
		dataL = dataset[nonzero(dataset[:,feaNum] < thres)[0],:]
		dataR = dataset[nonzero(dataset[:,feaNum] >= thres)[0],:]
		return dataL, dataR
	
	# 计算总的方差
	def getAllVar(self, dataset):
		return var(dataset[:,-1]) * shape(dataset)[0]
			
	def findFeatureAndThresParallel(self, feature, dataset, lock):
		# 每个样本遍历特征 i
		m,n = shape(dataset)
		
		dataset_t = dataset[np.lexsort(dataset[:,feature].T)]
		thresList = dataset_t[0][:,feature].T

		record = set()
		sum = 0.0
		sq_sum = 0.0
		sum_List = []
		sq_sum_List = []
		
		'''
		for i in range(m):
			number = dataset_t[0][i,-1]
			sum = sum + number
			sq_sum = sq_sum + number**2
			if i == 0 :
				sum_List.append(number)
				sq_sum_List.append(number**2)
			else:
				sum_List.append(number + sum_List[i-1])
				sq_sum_List.append(number**2 + sq_sum_List[i-1])
		'''
		sum_List = np.cumsum(dataset_t[0][:,-1]).tolist()[0]
		sq_sum_List = np.cumsum(np.square(dataset_t[0][:,-1])).tolist()[0]
		sum = sum_List[-1]
		sq_sum = sq_sum_List[-1]
		
		temp_Err = inf
		temp_bFea = -1
		temp_bThre = float("-inf")
		
		for index in range(m):
			if(thresList[0,index] in record):
				continue
			else:
				record.add(thresList[0,index])
			left_size = index
			right_size = m-index
			if left_size < max(int(floor(0.05*m)), 100) or right_size < max(int(floor(0.05*m)), 100):
				continue
			left_sum = sum_List[left_size-1]
			right_sum = sum - left_sum
			left_sq_sum = sq_sum_List[left_size-1]
			right_sq_sum = sq_sum - left_sq_sum

			var_left = left_sq_sum / left_size - (left_sum / left_size) ** 2
			var_right = right_sq_sum / right_size - (right_sum / right_size) ** 2
			
			tempError = var_left * left_size + var_right * right_size
			if tempError < temp_Err:
				temp_Err = tempError
				temp_bFea = feature
				temp_bThre = thresList[0,index]
		lock.acquire()
		if temp_Err < self.lowError:
			self.lowError = temp_Err
			self.bestFeature = temp_bFea
			self.bestThres = temp_bThre
		lock.release()

	def chooseBestFeature(self, dataset, featureList, max_depth):
		# 停止条件 1：标签相同
		if len(set(dataset[:,-1].T.tolist()[0])) == 1 or max_depth == 0:
			regLeaf = mean(dataset[:,-1])
			return None, regLeaf 
		
		# 停止条件 2：已完成所有标签分类
		if len(featureList) == 1:
			regLeaf = mean(dataset[:,-1])
			return None, regLeaf 
		
		m,n = shape(dataset)
		totalVar = self.getAllVar(dataset)
		self.bestFeature = -1
		self.bestThres = float('-inf')
		self.lowError = inf
		
		#begin = time.time()
		# 遍历剩余划分特征 i
		lock = threading.Lock()
		#Parallel(n_jobs=len(featureList),backend='threading')(delayed(self.findFeatureAndThresParallel)(feature, dataset, lock) for feature in featureList)
		#for feature in featureList:
			#self.findFeatureAndThresParallel(feature, dataset)
		t_List = []
		for feature in featureList:
			t_List.append(threading.Thread(self.findFeatureAndThresParallel, args=(feature, dataset, lock)))
		for t in t_List:
			t.start()
		for t in t_List:
			t.join()
		#end = time.time()
		#print("loop time:", end-begin)
		# 停止条件3：划分后方差更大，则取消划分
		if totalVar - self.lowError < self.varThres:
			return None, mean(dataset[:,-1])
		
		# 停止条件4：划分后数据集太小
		dataL, dataR = self.datasetSplit(dataset, self.bestFeature, self.bestThres)
		if shape(dataL)[0] < max(int(floor(0.05*m)), 100) or shape(dataR)[0] < max(int(floor(0.05*m)), 100):
			return None, mean(dataset[:,-1])
			
		# 成功则返回最佳特征和最小方差
		return self.bestFeature, self.bestThres
		
	# dataset: 数据集, featureList: 随机特征
	def createTree(self, dataset, max_depth=100):
		m, n = shape(dataset)
		#featureList = np.random.choice(range(n-1), 2, replace=False).tolist()
		featureList = list(range(n-1))
		bestFeat, bestThres = self.chooseBestFeature(dataset, featureList, max_depth) #最耗时
		
		if bestFeat == None:
			return bestThres
		regTree = {}
		# 记录此特征及阈值
		regTree['spliteIndex'] = bestFeat
		regTree['spliteValue'] = bestThres
		# 划分数据集
		dataL,dataR = self.datasetSplit(dataset, bestFeat, bestThres)
		regTree['left'] = self.createTree(dataL, max_depth-1)
		regTree['right'] = self.createTree(dataR, max_depth-1)
		return regTree
	
	def isTree(self, tree):
		return type(tree).__name__=='dict'
	
	def predict(self, tree, testData):
		if not self.isTree(tree):
			return float(tree)
		if testData[tree['spliteIndex']] < tree['spliteValue']:
			if not self.isTree(tree['left']):
				return float(tree['left'])
			else:
				return self.predict(tree['left'], testData)
		else:
			if not self.isTree(tree['right']):
				return float(tree['right'])
			else:
				return self.predict(tree['right'], testData)

class RandomForest:
	def __init__(self, n):
		self.treeNum = n
		self.treeList = []
		self.ct = CART_tree()

	def fit(self, dataset, jobs=1):
		'''
		Parallel(n_jobs=jobs)(delayed(self.parallelCreateTree)(i, dataset) for i in range(self.treeNum))
		print(self.treeList[0])
		'''
		m, n = shape(dataset)
		pool = multiprocessing.Pool(processes = jobs)
		#s = multiprocessing.Semaphore(4)
		for i in range(self.treeNum):
			#data_t = np.random.choice(range(m), 10000).tolist()
			#fea_t = np.random.choice(range(n-1), 2, replace=False).tolist()
			data_t = list(range(1000000))
			random_dataset = dataset[data_t,:]
			tt = createTreeThread(random_dataset)
			self.treeList.append(pool.apply_async(tt.run))
		pool.close()
		pool.join()
		for treeNum in range(len(self.treeList)):
			self.treeList[treeNum] = self.treeList[treeNum].get()
		#self.treeList.append(tree)

	def parallelCreateTree(self, i, dataset):
		m, n = shape(dataset)
		data_t = np.random.choice(range(m), m).tolist()
		fea_t = np.random.choice(range(n-1), 4, replace=False).tolist()
		random_dataset = dataset[data_t,:]
		tree = self.ct.createTree(random_dataset, fea_t) #最耗时
		print(tree)
		self.treeList.append(tree)
		
	def writeToFile(self, tree, index):
		res = json.dumps(tree)
		if not os.path.exists("./model"):
			os.mkdir("./model")
		with open("./model/tree_{}.json".format(index), "w") as f:
			f.write(res)
	
	def loadFromFile(self):
		for root, dirs, files in os.walk("./model"):
			for file in files:
				with open(file, "r") as f:
					for line in f.readlines():
						self.treeList.append(json.loads(line))
						
	def predict(self, testData):
		result = []
		for i in range(len(testData)):
			res = []
			for tree in self.treeList:
				res.append(self.ct.predict(tree, testData[i]))
			result.append(res)
		print(result)
		return np.matrix(result).mean(1).T.tolist()

class createTreeThread:
	def __init__(self, dataset):
		self.data = dataset
		self.ct = CART_tree()
		
	def run(self):
		self.tree = self.ct.createTree(self.data)
		return self.tree

def readDataset():
	trainSet = []
	labelSet = []
	for i in range(1, 2):
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
	'''
	trainData = pd.read_csv(os.path.join('data/', 'train1.csv'), header=None, \
		delimiter="\t", quoting=3)
	labelData = pd.read_csv(os.path.join('data/', 'label1.csv'), header=None, \
		delimiter="\t", quoting=3)
	for example in list(trainData[0]):
		cur_example = example.strip().split(',')
		fin_example = map(float, cur_example)
		trainSet.append(list(fin_example))
	for label in list(labelData[0]):
		labelSet.append(float(label))
	'''
	tS_matrix = np.matrix(trainSet)
	tL_matrix = np.matrix(labelSet)
	final_trainSet = np.insert(tS_matrix, 13, values=tL_matrix, axis=1)
	
	return final_trainSet

def readTestData():
	testSet = []
	for i in range(1, 7):
		testData = pd.read_csv(os.path.join('data/', 'test{}.csv'.format(i)), header=None, \
			delimiter="\t", quoting=3)
		for example in list(testData[0]):
			cur_example = example.strip().split(',')
			fin_example = map(float, cur_example)
			testset.append(list(fin_example))
	'''
	testData = pd.read_csv(os.path.join('data/', 'test1.csv'), header=None, \
		delimiter="\t", quoting=3)
	for example in list(testData[0]):
		cur_example = example.strip().split(',')
		fin_example = map(float, cur_example)
		testSet.append(list(fin_example))
	'''
	return testSet
	
if __name__=="__main__":
	print("read dataset")
	trainData = readDataset()
	print("begin generate forest")
	begin = time.time()
	rf = RandomForest(1)
	rf.fit(trainData, jobs=4)
	end = time.time()
	print(end-begin)
	
	#test = readTestData()
	#print(test[0])
	
	r = rf.predict([[-0.22123,-218,-15,151,-1,-74,-65,0.4587,0.46732,-0.47841,0.99324,0.48148,-0.32254],\
				[-0.50568,2,-218,-15,151,-1,-74,-0.49038,0.4587,0.46732,-0.47841,0.99324,0.48148],\
				[-0.073529,-10.5,2,-218,-15,151,-1,0.43519,-0.49038,0.4587,0.46732,-0.47841,0.99324]])
	print(r)