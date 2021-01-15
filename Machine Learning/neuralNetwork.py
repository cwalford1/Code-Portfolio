import numpy as np
import pandas as pd
import mLFuncs
import os

"""Here I'll attempt to write up an object that can initialize a simple neural network with arbitrary number of layers and arbitrary number of nodes in intermediate layers.
This isn't a particularly practical thing to do, but the goal is that this project forces me to become deeply familiar with the inner mechanisms of neural networks,
particularly the calculus and linear algebra of backpropagation.

Post Notes:
I made a few design choices to simplify the process, and while in hindsight these definitely limited the functionality/accuracy of my networks.  However, this project already stretched how long I wanted to work on it,
and the added complexity would certaintly have exaserbated this problem.  I don't regret these choices, but they're certainly things I'll change when I revisit this exercise.

First, I decided to make all the intermediate layers have equal number of nodes as this means I only have to adjust the shape of weight matrices for the input and output layer.
Second, I wrote the network to take single labelled inputs at a time rather than an array of several labelled inputs.  I did this to make it easier to conceptually follow an input through feedforward and backpropagation,
but this will definitely decrease time efficiency and might reduce accuracy as the cost function landscape is only a single input rather than the sum of several inputs. Most of the calculations are linear so I didn't think this would matter
(and I'm still not certain if it does) but it's still a limitation. 
Finally, I did not implement biases in my feedforward process (or perhaps I assumed all biases to be zero).  This was done to lighten the backpropogation calculus as I was doing a lot of it analytically by hand, but limits the potential accuracy of the network.

I plan to make a LaTex write up of the mathematics behind this program as math like analytically solving for expressions for derivatives of weights
was the majority of the work that went into this project, so check the directory for that."""

# functions and their derivatives for use in neural net
# using sigmoid for activation function and sum of squared errors for cost function

def sigmoid(x):
    return 1/(1+np.power(np.e,-x))

def dSigmoid(x):
    return ((np.power(np.e,-x))/(np.power(1+np.power(np.e,-x),2)))

def inverseSigmoid(x):
    return np.log(x/(1-x))

def sumSquaredErrors(y,yHat):
    return np.square(y-yHat)/len(y)

def dSumSquaredErrors(y,yHat):
    return 2*(y-yHat)/len(y)


class neuralNetwork:
    def __init__(self,x,y,nLayers=2,nNodes=3,lossFunction=sumSquaredErrors):
        #initialize a bunch of matrices the number and shape of which are determined by the shape of input/output layers and 
        self.input=np.reshape(x,(len(x),1))
        self.x=x
        self.nLayers=nLayers
        self.nNodes=nNodes
        self.weights=[]
        self.y=y
        self.firstWeight=np.random.rand(nNodes,len(self.input))
        self.weights.append(self.firstWeight)
        for i in range(nLayers-1):
            self.weights.append(np.random.rand(nNodes,nNodes))
        self.lastWeight=np.random.rand(len(self.y),nNodes)
        self.weights.append(self.lastWeight)

    def feedForward(self,activationFunction=sigmoid):
        #perform feedforward algorithm
        #I refer to "layers" as the vector of values of weighted sums at nodes before applying the activatin function
        #and "activations" as the vector values corresponding to applying the activation function to the layers vector "
        self.layers=[]
        self.activations=[]
        self.activations.append(self.x)
        for i,weight in enumerate(self.weights):
            if i==0:
                self.layers.append(np.dot(weight,self.x))
                self.activations.append((activationFunction(self.layers[-1])))
            else:
                self.layers.append(np.dot(weight,self.activations[-1]))
                self.activations.append(activationFunction(self.layers[-1]))

    def backProp(self,dActivationFunction,dCostFunction,activationFunction):
        #Perform backPropagation by calculating derivative of each weight with respect to the cost function
        self.dWeightsReversed=[]
        self.dAdCostReversed=[]
        #the first (technically last) weight derivative matrix is calculated seperately as the backpropagation process is iterative so we need a starting point.
        dAdCost=dCostFunction(activationFunction(self.y),self.activations[-1])
        self.dAdCostReversed.append(dAdCost)
        dLdC=dAdCost*dSigmoid(self.layers[-1])
        dLdCT=np.array([dLdC]).T
        self.dWeightsReversed.append(dLdCT*self.activations[-2])
        self.weights[-1]+=self.dWeightsReversed[0]

        for i in range(len(self.weights[:-1])):
            #we're itterating through the weights and activations lists backwards so we need a backward counting index variable
            i=-i-1
            #derivative of this activation (A) with respect to previous activation (Ap)
            #note that we're looping through activations backwards so the "previous" activation is a higher index activation
            dAdC=np.dot(self.weights[i].T,dActivationFunction(np.array([self.layers[i]]).T)*self.dAdCostReversed[-1])
            self.dAdCostReversed.append(dAdC)
            dWdA=np.array([dActivationFunction(self.layers[i-1])]).T*self.activations[i-2]
            self.dWeightsReversed.append(dWdA)
            self.weights[i-1]+=self.dWeightsReversed[-1]

#A copy of the above class and functions can be found in mLFuncs for future use.
#to test the neural network I'll provide the truth table to a couple different logic gates and see if the nets can figure them out.

def RPD(x,y):
    return 2*(x-y)/(np.abs(x)+np.abs(y))

def testNet(inputs,labels,nTrials=1000,nLayers=1,nNodes=3):
    net=neuralNetwork(xs[0],ys[0],nLayers=nLayers,nNodes=nNodes)
    #train the net on the labelled dataset
    for i in range(nTrials):
        for x,y in zip(xs,ys):
            net.x=x
            net.y=y
            net.feedForward()
            net.backProp(dSigmoid,dSumSquaredErrors,sigmoid)
    #test the trained net on the dataset and display results
    #The result of the net is the last activation which is the output of a sigmoid function, so it's helpful to see both 
    #how close the inverse sigmoid of the activation is and how close the activation is to the sigmoid of the correct answer
    
    for x,y in zip(xs,ys):
        net.x=x
        net.y=y
        net.feedForward()
        print('\nResults for ({},{}).'.format(x[0],x[1]))

        realRes=inverseSigmoid(net.activations[-1])[0]
        realSol=y[0]
        sigRes=net.activations[-1][0]
        sigSol=sigmoid(y)[0]
        realErr=np.abs(realSol-realRes)
        yArray=np.array(ys)
        sigErr=np.abs(sigSol-sigRes)/(sigmoid(1)-sigmoid(0))

        print('\nReal Space: Final activation={:.2f}, correct answer={}, error={:.2f}.'.format(realRes,realSol,realErr))
        #Because the range between sig(0) and sig(1) is smaller than that between 1 and 0, I scale the sigmoid space error up to make it comparable to the real space error.
        #A helpful result of this scaling is that if the relative error is less than 0.5, then we immediately know result would round to the correct value.
        print('Sigmoid Space: Final activation={:.2f}, correct answer={:.2f}, relative error={:.2f}.\n'.format(sigRes,sigSol,sigErr))

#Here I test if the net can learn an AND gate.  I'll test the net on a few subsets of each truth tabel and then the entire table.

#AND first.
#training/testing 0,1 1,0 and 0,0
xs=[np.array([0,1]),np.array([1,0]),np.array([0,0])]
ys=[np.array([0]),np.array([0]),np.array([0])]
print('Training on (0,1), (1,0), and (0,0)...')
testNet(xs,ys)
print("""Here the net correctly converges to giving an activation of 0 for all inputs.  
This is a pretty low bar as it doesn't have to distinguish between inputs at all, but still a positive result.""")
input('\npres ret to continue...')

os.system('cls' if os.name=='nt' else 'clear')

# training/testing 0,0 and 1,1  
xs=[np.array([0,0]),np.array([1,1])]
ys=[np.array([0]),np.array([1])]
print('Training on (0,0) and (1,1)... ')
testNet(xs,ys)
print("""When various correct output values are introduced the net start to struggle, but it is definitely distinguishing between the inputs.
All in all, if we round to the nearest valid output (0 or 1 for real space and .5 or .73 for sigmoid space) the net is still batting 1000.""")
input('\npres ret to continue...')
os.system('cls' if os.name=='nt' else 'clear')

#training/testing on all
xs=[np.array([0,0]),np.array([1,0]),np.array([0,1]),np.array([1,1])]
ys=[np.array([0]),np.array([0]),np.array([0]),np.array([1])]
print('Training on whole truth table...')
testNet(xs,ys)
input('\npres ret to continue...')
os.system('cls' if os.name=='nt' else 'clear')
