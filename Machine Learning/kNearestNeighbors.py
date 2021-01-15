import pandas as pd 
import numpy as np
import math
from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 

"""About this task: I want to get a feel for the k nearest neighbor algorithm and experiment with the Iris dataset.
First I'm going to use the sklearn iris dataset and KNN objects to establish a baseline and get some experience using this tool (this will be largly a copy-paste job from IMLP).
However, given that KNN is a pretty simple algorithm and simply running some pre-written functions isn't particularly enlightening, 
I'll also write up my own version (which will be lacking any under-the-hood optimizations) to see how it compares.
"""

#gets the iris dataset bunch object
iris_dataset=load_iris()
#iris_dataset behaves like a dictionary, and the important keys are 'data' and 'target' as these hold an array of measurements of irises and the correct iris cultivar for these measurements respectively
#the iris cultivars are represented by a 0, 1, or 2 corresponding to setosa, versicolor, and virginica.

#extract the data from iris_dataset object and splits into test data and training data.
#default ratio of training data to total data is .75
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], shuffle=True)

#KNeighborsClassifier returns a k nearest neighbor object which we can pass our training data and will return predictions
knn = KNeighborsClassifier(n_neighbors=1)

#train our knn object on the training data
knn.fit(X_train, y_train)

#feed knn test data and see how well it does
y_pred = knn.predict(X_test)

print("Sklearn default KNN test score: {:.2f}".format(np.mean(y_pred == y_test)))


#all right now I'll give it a crack with no help from sklearn
#Open another copy of the iris dataset. and split it into data and targets
data=pd.read_csv('iris.data').to_numpy()
stripped=data[:,:4]
flowers=data[:,-1]
percents=np.array([])

#randomly assign rows to be either training or test data
trainingPercent=0.75
indices=np.arange(0,len(data))
np.random.shuffle(indices)

trainingIndices=indices[:math.floor(len(indices)*trainingPercent)]
testIndices=indices[math.floor(len(indices)*trainingPercent):]

trainingData=stripped[trainingIndices]
testData=stripped[testIndices]
testLabels=flowers[testIndices]
trainingLabels=flowers[trainingIndices]


#initialize some variables to intermediate save data
nearestNeighbors=np.array([])
right=0
wrong=0

#calculate nearest neighbor for each element of training set
for i,test in enumerate(testData):
    for j,train in enumerate(trainingData):
        distance=np.sum(np.square(test-train))
        if j==0:
            min=distance
            index=j
        elif distance<min:
            min=distance
            index=j
        else:
            pass
    nearestNeighbors=np.append(nearestNeighbors,index)

    #check to see if nearest neighbor was of the same type as test data
    if testLabels[i]==trainingLabels[index]:
        right+=1
    else:
        wrong+=1

#calculate and print our success rate
tot=len(testIndices)
percentCorrect=right/tot*100
percents=np.append(percents,percentCorrect)
print('My nearest neighbor score: {}'.format(np.mean(percents)))

"""Closing thoughts:
It looks like my implamentation can achieve roughly the same results as the sklearn KNN object.  This is simplified version of KNN as I only find 1 nearest neighbor.  I have written up a full KNN function in mLFuncs.py.
I'm also interested in more robust comparisons as well as experimenting with free parameters like the value of K in kNN and the ratio of training to test data.
Further experimentation can be found in nearestNeighborTests.py"""
