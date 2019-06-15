import numpy as np
import csv, os
import pandas as pd
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

originSet = []
label = []
testSet = []

with open("./data/testSet.csv", "r") as f:
	f_csv = csv.reader(f)
	headers = next(f_csv)
	for row in f_csv:
		formatRow = [float(n) for n in row]
		testSet.append(formatRow)

with open("./data/trainSet.csv", "r") as f:
	f_csv = csv.reader(f)
	headers = next(f_csv)
	for row in f_csv:
		formatRow = [float(n) for n in row]
		originSet.append(formatRow)
		
temp = np.matrix(originSet)
label = temp[:,32].tolist()
train_data = temp[:, 0:32].tolist()

scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
testSet = scaler.transform(testSet)

'''
clf = LinearSVC() # 创建线性可分svm模型，参数均使用默认值
clf.fit(train_data, label)  # 训练模型
result = clf.predict(testSet)  # 使用模型预测值
#print('预测结果：',result)
'''

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8, 4), activation='logistic',
                solver='adam', learning_rate_init=0.001, max_iter=200, verbose = True)

mlp.fit(train_data, label)

result = mlp.predict(testSet)
print(result)
r = []
for i in range(len(testSet)):
	r.append(i+1)
result = [int(n) for n in result]
output = pd.DataFrame( data={"id": r, "Predicted":result} )

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'result.csv'), index=False, quoting=3)

