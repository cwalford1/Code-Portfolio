import pandas as pd 
import numpy as np
import math

#functions and their derivatives for use in neural net
#using sigmoid for activatio nfunction and sum of squared errors for cost function
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

def kNearestNeighbors(data, labels, testData=None, trainingData=None,trainingLabels=None,testLabels=None, mode='mixed',labelled=True,trainingPercent=0.75,k=1):
    if mode=='mixed':
        #randomly assign data to test or training
        indices=np.arange(0,len(data))
        np.random.shuffle(indices)

        trainingIndices=indices[:math.floor(len(indices)*trainingPercent)]
        testIndices=indices[math.floor(len(indices)*trainingPercent):]

        trainingData=data[trainingIndices]
        testData=data[testIndices]
        testLabels=labels[testIndices]
        trainingLabels=labels[trainingIndices] 
    else:
        pass

    nearestNeighbors=np.array([])
    right=0
    wrong=0

    #calculate nearest neighbors for each element of training set
    for i,test in enumerate(testData):
        mins=np.array([])
        indices=np.array([])
        for j,train in enumerate(trainingData):
            distance=np.sum(np.square(test-train))
            if j<k:
                mins=np.append(mins,distance)
                indices=np.append(indices,j)
            elif distance<max(mins):
                #remove the largest element of mins from mins and indices and replace with new distance&index
                mins=np.delete(mins,np.where(mins==max(mins))[0][0])
                mins=np.append(mins,distance)
                indices=np.delete(indices,np.where(mins==max(mins))[0][0])
                indices=np.append(indices,j)
            else:
                pass
        
        candidates=[]
        tieBreakerInfo={}
        votes=np.array([])

        #the prediction label is whichever label was associated with a plurality of k nearest neighbors
        #here I determine the winner through an 'election' where each of the k nearest neighbors 'votes' for their associated label
        
        #loop through the indices of the k nearest neighbors
        for x,index in enumerate(indices):
            index=int(index)
            #the candidate this neighbor votes for is the label associated with the same index
            candidate=trainingLabels[index]
            #this first if is redundant with the following elif, but numpy throws a warning if you attempt to check if a string is in an empty array and I'd like to avoid this
            if x==0:
                #we keep track of each candidates minimum distance voter for tiebreakers later
                tieBreakerInfo[candidate]=mins[x]
                #keep track of what candidates are on the board
                candidates.append(candidate)
                #count a vote
                votes=np.append(votes,1)
            #votes are associated with candidates through index in the two arrays.  candidate i from the candidates array has votes equal to the ith element of the votes array

            #if this candidate hasn't isn't on the board yet (could include the above case)
            elif candidate not in candidates:
                tieBreakerInfo[candidate]=mins[x]
                candidates.append(candidate)
                votes=np.append(votes,1)
            else:
                #if this candidate is on the board we add one to its vote
                votes[candidates.index(candidate)]+=1
                #check to see if this voter has a lower minimum distance than previous voters.  If so, replace the mimimum distance vote associated with this candidate
                if tieBreakerInfo[candidate]>mins[x]:
                    tieBreakerInfo[candidate]=mins[x]
                else:
                    pass
        #if there is not a tie we record the prediciton and move on
        if len(np.where(votes==max(votes))[0])==1:
            winner=candidates[np.where(votes==max(votes))[0][0]]
            nearestNeighbors=np.append(nearestNeighbors,winner)

        #Ties go to the candidate with the lowest minimum distance voter recorded in tieBreakerInfo
        else:
            ties=np.array([])
            #collect the candidates that had the highest number of votes
            for loc in np.where(votes==max(votes))[0]:
                ties=np.append(ties,candidates[loc])
            #go through these second phase candidates and see who has the lowest minimum distance voter
            for count,tie in enumerate(ties):
                if count==0:
                    minTieDistance=tieBreakerInfo[tie]
                    winner=tie
                elif minTieDistance<tieBreakerInfo[tie]:
                    minTieDistance=tieBreakerInfo[tie]
                    winner=tie
                else:
                    pass

        #the winner of the election is our prediction label which we record
        nearestNeighbors=np.append(nearestNeighbors,winner)

        #check to see if nearest neighbor was of the same type as test data
        if testLabels[i]==winner:
            right+=1
        else:
            wrong+=1

    #calculate success rate
    tot=len(testIndices)
    percentCorrect=right/tot*100

    return nearestNeighbors,percentCorrect

