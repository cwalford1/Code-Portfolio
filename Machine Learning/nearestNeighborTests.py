import pandas as pd 
import numpy as np
import math
from mLFuncs import kNearestNeighbors 
from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from timeit import timeit
import matplotlib.pyplot as plt

"""About this task:
Here I perform several tests on and comparisons of my KNN algorithm and that included in sklearn including:
-Timing experiment
-accuracy vs training data size
-accuracy vs k.

A summary of results can be found at the bottom of this file and copies of any grpahs produced can be found in the plots subdirectory.
"""

trials=100

#timing experiment:
#run my kNearestNeighbors function and that from sklearn 100 times to produce an average time per run to establish which is more efficient
print('performing time comparison... \n')
time1=timeit("""from mLFuncs import kNearestNeighbors
import pandas as pd
data=pd.read_csv('iris.data').to_numpy()
stripped=data[:,:4]
flowers=data[:,-1]
results=kNearestNeighbors(stripped,flowers,k=1)""",number=trials)

time2=timeit("""from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

iris_dataset=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], shuffle=True)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
""",number=trials)

print('\nMy nearest neighbor function average runtime: {:.4f} seconds'.format(time1/trials))
print('sklearn nearest neighbor average runtime: {:.4f} seconds'.format(time2/trials))
print('My nearest neighbor was {:.2f} times slower than the sklearn implementation.'.format(time1/time2))
#We see that my implementation is significantly slower than the sklearn kNN function.  This is expected as almost no thought went into making my function computationally optimal
#while I assume the sklearn implementation is highly optimized.
input('\npres ret to continue ')

data=pd.read_csv('iris.data').to_numpy()
stripped=data[:,:4]
flowers=data[:,-1]

#Accuracy vs training data size experiment:
#Here I'll test how informationally efficient each KNN algorithm is by ramping down the ammount of the dataset reserved for training data.

iris_dataset=load_iris()
#create a range of data ratios to test with from 25% test data (default) to 99% test data.
testRatio=np.arange(25,100)/100
myAccuracies=np.array([])
sklearnAccuracies=np.array([])
#loop over data ratios and record accuracy of both KNN methods.  10 trials for each ratio as accuracy fluctuates slightly.
print('plotting accuracy vs training dataset size... \n')
for i in range(10):
    for item in testRatio:
        results=kNearestNeighbors(iris_dataset['data'],iris_dataset['target'],k=1,trainingPercent=1-item)
        myAccuracies=np.append(myAccuracies,results[1])

        testSize=math.floor(item*len(iris_dataset['data']))
        trainSize=len(iris_dataset['data'])-testSize
        X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], shuffle=True,test_size=int(testSize),train_size=int(trainSize))
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        sklearnAccuracies=np.append(sklearnAccuracies,np.mean(y_pred == y_test))
    print('{}% '.format((i+1)*10))

#reshape and condense the two lists holding results from all 10 trials of every tested ratio to be single average results from the ratios.
myAccuracies=myAccuracies.reshape(10,len(testRatio))
myAccuracies=myAccuracies.T
myAccuracies=np.mean(myAccuracies,axis=1)

sklearnAccuracies=sklearnAccuracies.reshape(10,len(testRatio))
sklearnAccuracies=sklearnAccuracies.T
sklearnAccuracies=np.mean(sklearnAccuracies,axis=1)*100


#plot results
fig,ax=plt.subplots(1)
ax.set_title('Accuracy of KNNs with throttled training data')
ax.set_ylabel('% Accuracy of KNN averaged over 10 trials')
ax.set_xlabel('Fraction of data reserved for testing')
ax.plot(testRatio,myAccuracies,label='My KNN')
ax.plot(testRatio,sklearnAccuracies,label='Sklearn KNN')
ax.legend()
plt.show()
print('\nplotting accuracy vs K value...\n')
myAccuracies=np.array([]) 
sklearnAccuracies=np.array([]) 
ks=np.arange(1,11)
for i in range(10):
    for k in ks:
        results=kNearestNeighbors(iris_dataset['data'],iris_dataset['target'],k=k)
        myAccuracies=np.append(myAccuracies,results[1])

        X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], shuffle=True)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        sklearnAccuracies=np.append(sklearnAccuracies,np.mean(y_pred == y_test))
    print('{}%'.format((i+1)*10))

myAccuracies=myAccuracies.reshape(10,len(ks))
myAccuracies=myAccuracies.T
myAccuracies=np.mean(myAccuracies,axis=1)

sklearnAccuracies=sklearnAccuracies.reshape(10,len(ks))
sklearnAccuracies=sklearnAccuracies.T
sklearnAccuracies=np.mean(sklearnAccuracies,axis=1)*100

fig,ax=plt.subplots(1) 
ax.set_title('accuracy vs k value')
ax.set_ylabel('% Accuracy of KNN averaged over 10 trials')
ax.set_xlabel('k value')
ax.plot(ks,myAccuracies,label='My KNN')
ax.plot(ks,sklearnAccuracies,label='sklearn KNN')
ax.legend()
plt.show()
