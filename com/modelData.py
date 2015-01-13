import numpy as np
import pandas as pd
import os
import scipy.spatial.distance as ssd
import collections
import time
from random import randint
from matplotlib import pyplot as pp
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from patsy.highlevel import dmatrices
from sklearn.cross_validation import StratifiedShuffleSplit as SSS
from sklearn import linear_model
    

if __name__ == '__main__':
    pass


startTime = int(round(time.time() * 1000))

# Set pointer to correct destination
basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "wine.data"))

# Parse 'wine.data' file into two variables x, y -> (x = labels, y = feature data).
x = np.loadtxt(filepath, delimiter=',')
y = np.loadtxt(filepath, delimiter=',', usecols=range(0,1))

# Label each feature for sake of clarity. Used in creating plot views.
labels = ['Class','Alcohol', 'Malic_acid', 'Ash', 'Alcality_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
            'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines', 'Proline']


def linearRegression():
    # Task 1a) 1
    df = pd.DataFrame(x,columns=labels)
    y, X = dmatrices(generateLabels(), df, return_type='matrix')
    y = np.ravel(y)
    regressionmodel = createRegressionModel(X, y)
    errorCount =  inferErrors(regressionmodel, X, y)
    print 'Errors in whole data: ' + str(errorCount)
    # End Task 1a) 1
    
    # Task 1a) 2
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=np.random)
    
    regressionModel = createRegressionModel(xTrain, yTrain)
    trainErrorCount = inferErrors(regressionModel, xTrain, yTrain)
    print 'Errors in randomized training data (3/4): ' + str(trainErrorCount)
    
    testErrorCount = inferErrors(regressionModel, xTest, yTest)
    print 'Errors in randomized test data (1/4): ' + str(testErrorCount)
    # End Task 1a) 2
    
    numOfTrainErrors, numOfTestErrors = tenFoldExperiment(X, y)
    createBoxPlot(numOfTrainErrors, numOfTestErrors)
    calculateOptimalRegParam(xTrain, yTrain, xTest, yTest, True)


# Task 1a) 3
def tenFoldExperiment(X, y):
    stratifiedShuffleSplit = SSS(y,10,test_size=0.25,random_state=np.random)
    i = 0
    #trainMatrix = []
    #testMatrix = []
    numOfTrainErrors = []
    numOfTestErrors = []
    for trainIndex, testIndex in stratifiedShuffleSplit:
        i = i+1
        xTwoTrain, xTwoTest = X[trainIndex],X[testIndex]
        yTwoTrain,yTwoTest = y[trainIndex],y[testIndex]
        
        regressionModel = createRegressionModel(xTwoTrain, yTwoTrain)
        numOfErrorsTrain = inferErrors(regressionModel, xTwoTrain, yTwoTrain)
        numOfTrainErrors.append(numOfErrorsTrain)        
        
        numOfErrorsTest = inferErrors(regressionModel, xTwoTest, yTwoTest)
        numOfTestErrors.append(numOfErrorsTest)
        
        # Calculating OPT reg param for each train and test data. (remove first comment sign to test)
        '''trainErrors, testErrors = calculateOptimalRegParam(xTwoTrain, yTwoTrain, xTwoTest, yTwoTest, False)
        trainMatrix.append(trainErrors)
        testMatrix.append(testErrors) 
    
    # 1b) 1
    trainMatrix = np.matrix(trainMatrix)
    testMatrix = np.matrix(testMatrix)
    pp.plot(trainMatrix)
    pp.plot(testMatrix)
    pp.show()'''
       
    numOfTrainErrors = np.array(numOfTrainErrors)
    numOfTestErrors = np.array(numOfTestErrors)
    
    print
    print 'Errors in training data sets (10-fold):'
    for i in range(0, len(numOfTrainErrors)):
        print 'Set ' + str(i + 1) + ': ' + str(numOfTrainErrors[i])
    
    print 
    print 'Std deviation of train data: ' + str(numOfTrainErrors.std())
    print 'Mean of training data: ' + str(numOfTrainErrors.mean())   
    
    print
    print 'Errors in test data sets (10-fold):'
    for i in range(0, len(numOfTestErrors)):
        print 'Set ' + str(i + 1) + ': ' + str(numOfTestErrors[i])
         
    print
    print 'Std deviation of test data: ' + str(numOfTestErrors.std())     
    print 'Mean of test data: ' +str(numOfTestErrors.mean())
    
    return numOfTrainErrors, numOfTestErrors


# Task 1b)
def calculateOptimalRegParam(xTrain, yTrain, xTest, yTest, doPlot):
    alphas = np.logspace(-5, 5.0, 178)
    enet = linear_model.ElasticNet(l1_ratio=0.5)
    trainErrors = list()
    testErrors = list()
    for alpha in alphas:
        enet.set_params(alpha=alpha)
        enet.fit(xTrain, yTrain)
        trainErrors.append(enet.score(xTrain, yTrain))
        testErrors.append(enet.score(xTest, yTest))
    
    maxTrainErrorIndex = np.argmax(trainErrors)
    maxTestErrorIndex = np.argmax(testErrors)
    optimalTrainRegParam = alphas[maxTrainErrorIndex]
    optimalTestRegParam = alphas[maxTestErrorIndex]
    print
    print("Optimal train regularization parameter: %s" % optimalTrainRegParam)
    print ("Error == %s" % trainErrors[maxTrainErrorIndex])
    print("Optimal test regularization parameter: %s" % optimalTestRegParam)
    print ("Error == %s" % testErrors[maxTestErrorIndex])
      
    if (doPlot):
        plotErrorsWithOptRegParam(alphas, trainErrors, testErrors)
        
    return trainErrors, testErrors
    

def plotErrorsWithOptRegParam(alphas, trainErrors, testErrors):
    pp.subplot(2, 1, 1)
    pp.semilogx(alphas, trainErrors, label='Train')
    pp.semilogx(alphas, testErrors, label='Test')
    pp.legend(loc='lower left')
    pp.ylim([0, 1.2])
    pp.xlabel('Regularization parameter')
    pp.ylabel('Performance')
    pp.show()
    
    
def calculateErrorCount(clazz,predictedValues):
    arr = np.array(filter(None,[predictedValuese if clazze != predictedValuese else '' for clazze, predictedValuese in zip(clazz,predictedValues)]))
    return len(arr)
 

def inferErrors(regressionModel, xData, yData):
    predictedY = regressionModel.predict(xData)
    return calculateErrorCount(yData, predictedY)


def createRegressionModel(xTrain, yTrain):
    regressionModel = LogisticRegression()
    regressionModel = regressionModel.fit(xTrain,yTrain);
    return regressionModel


# Task 1a) 4
def createBoxPlot(trainErrCount, testErrCount):
    data = [trainErrCount, testErrCount]
    colors = ['cyan','pink']
    pp.figure()
    box = pp.boxplot(data, labels=['Training','Test'], patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    pp.show()


def knn(k, dtrain, dtest, dtr_label):
    pred_class = []
    for _, di in enumerate(dtest):
        distances = []
        for ij, dj in enumerate(dtrain):
            distances.append((ssd.euclidean(di,dj), ij))
        k_nn = sorted(distances)[:k]
        pred_class.append(classify(k_nn, dtr_label))

    return pred_class


def classify(k_nn, dtr_label):    
    dlabel = []
    for _, idx in k_nn:
        dlabel.append(dtr_label[idx])

    return np.argmax(np.bincount(dlabel))


def evaluate(result):
    eval_result = np.zeros(2,int)
    for x in result:
        if x == 0:
            eval_result[0] += 1
        else:
            eval_result[1] += 1
    return eval_result


def NNC():
    printNNCHeader()
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.5)
    foo(yTest)
    print 'Train error percent from whole dataset:'
    calcNNError(xTrain,xTest,yTrain,len(xTrain), False)
    calcTenFoldError(xTrain, xTest, yTrain)
    plotKTrainAndTestError(xTrain,xTest,yTrain)
    print


def plotKTrainAndTestError(xTrain,xTest,yTrain):
    print 'Train errors (k = 1...30):'
    for k in range(1,31):
        fullresult = calcNNError(xTrain, xTest, yTrain, k, True)
    pp.plot(fullresult)
    pp.show()


def calcTenFoldError(xTrain,xTest,yTrain):   
    stratifiedShuffleSplit = SSS(y,10,test_size=0.25,random_state=np.random)
    for trainIdx, testIdx in stratifiedShuffleSplit:
        xTrain, xTest = x[trainIdx],x[testIdx]
        yTrain, yTest = y[trainIdx],y[testIdx]
        
        print
        print 'Train errors percentage (ten fold):'
        calcNNError(xTrain, xTest, yTrain, len(xTrain), False)
        
        print
        print 'Test errors percentage (ten fold):'
        calcNNError(xTest, xTrain, yTest, len(xTest), False)
        print
        break
     

def calcNNError(xTrain, xTest, yTrain, count, isK):
    fullResult = []
    correct = 0
    false = 0
    results = []
    
    if (count == 1):
        pred_class = knn(count, xTrain, xTest, yTrain)
        eval_result = evaluate(pred_class-yTrain[count])
        results.append(eval_result[0])
        results.append(eval_result[1])

        correct += results[0]
        false += results[1]
        fullResult.append(results)
        results = []   
    else:    
        for i in range(1, count):
            pred_class = knn(i, xTrain, xTest, yTrain)
            eval_result = evaluate(pred_class-yTrain[i])
            results.append(eval_result[0])
            results.append(eval_result[1])
    
            correct += results[0]
            false += results[1]
            fullResult.append(results)
            results = []
    
    if (isK == True):
        print 'K = {index}: {percentage}'.format(index=str(count), percentage=str(float(false)/float(correct+false) * 100.0))
    else: 
        print str(float(false)/float(correct+false) * 100.0) + '%'
        
    return fullResult


def tenTimesKMeansCluster():
    lastCentroids = []
    isFalse = True
    for _ in range(10):
        centroids, labels = KMeansClustering(randint(1,100))
        if lastCentroids != []:
            if (compareCentroids(centroids,lastCentroids) == False):
                isFalse = False
                break
        lastCentroids = centroids
    if (isFalse == True):
        print 'Result: All centroids were same'
    else:
        print 'Result: Not all centroids were the same'


def compareCentroids(cents, lastCents):
    for k in range(len(cents)):
        if collections.Counter(cents[k]) == collections.Counter(lastCents[k]):
            continue
        else:
            return False
            break
        
    return True
    

def KMeansClustering(numOfInit):
    km = KMeans(n_clusters = 3, n_init = numOfInit)
    km.fit(x)
    centroids = km.cluster_centers_
    labels = km.labels_
    return centroids, labels


def calcCentroidsAndClusterLabels():
    centroids, labels = KMeansClustering(10)
    print 'Centroids:'
    print centroids
    labelCounts = [0,0,0]
    for i in range(len(labels)):
        labelCounts[labels[i]] += 1
    clusterLabelIndex = np.argmax(labelCounts)
    classificationError = 0
    for k in range(len(labelCounts)):
        if k != clusterLabelIndex:
            classificationError += labelCounts[k]
    print
    print '"Classification errors": %s' % str(classificationError)
    print


def KMeansMain():
    calcCentroidsAndClusterLabels()
    tenTimesKMeansCluster()


def foo(yTest):
    return


def generateLabels():
    return str(labels[0]) + '~' + "+".join(labels[1:])


def getRunningTime():
    print '------------------------------------------------'
    print
    print 'Total running time: ' + str((int(round(time.time() * 1000)) - startTime) / 1000) + '(s)' 


def printLMAndROHeader():
    print '------------------------------------------------'
    print
    print '# Assignment 1: Linear modeling and optimization parameter'
    print


def printNNCHeader():
    print '------------------------------------------------'
    print
    print '# Assignment 2: Nearest neighbor classification'
    print


def printKMeanHeader():
    print '------------------------------------------------'
    print
    print '# Assignment 3: KMean clustering'
    print

   
linearRegression()    
NNC()
KMeansMain()
getRunningTime()