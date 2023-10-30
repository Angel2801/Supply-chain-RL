from data.genset import generateDataSet
from data.process import divideData

dataset = generateDataSet()
X_train, X_test = divideData(dataset)
print(type(X_train[:,2:3].flatten()))