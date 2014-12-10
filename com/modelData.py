import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as pp
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from patsy.highlevel import dmatrices
from sklearn.cross_validation import StratifiedShuffleSplit as SSS

if __name__ == '__main__':
    pass

# Set pointer to correct destination
basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "wine.data"))

# Parse 'wine.data' file into two variables x, y -> (x = labels, y = feature data).
x = np.loadtxt(filepath, delimiter=',')
y = np.loadtxt(filepath, delimiter=',', usecols=range(0,1))

# Label each feature for sake of clarity. Used in creating plot views.
labels = ['Class','Alcohol', 'Malic_acid', 'Ash', 'Alcality_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
            'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines', 'Proline']


def linearRegression(matrix):
    df = pd.DataFrame(matrix,columns=labels)
    y, X = dmatrices(generateLabels(), df, return_type='matrix')
    y = np.ravel(y)
    errorCount =  inferErrors(X, y, X, y)
    print 'Errors in whole data: ' + str(errorCount)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=np.random)
    trainErrorCount = inferErrors(X_train, y_train, X_train, y_train)
    print 'Errors in randomized training data (3/4): ' + str(trainErrorCount)
    
    testErrorCount = inferErrors(X_train, y_train, X_test, y_test)
    print 'Errors in randomized test data (1/4): ' + str(testErrorCount)
    
    stratifiedShuffleSplit = SSS(y,10,test_size=0.25,random_state=np.random)
    i = 0
    numOfTrainErrors = []
    numOfTestErrors = []
    for train_index, test_index in stratifiedShuffleSplit:
        i = i+1
        X_2train, X_2test = X[train_index],X[test_index]
        y_2train,y_2test = y[train_index],y[test_index]
        numOfErrorsTrain = inferErrors(X_2train, y_2train, X_2train, y_2train)
        numOfTrainErrors.append(numOfErrorsTrain)        
        
        numOfErrorsTest = inferErrors(X_2train, y_2train, X_2test, y_2test)
        numOfTestErrors.append(numOfErrorsTest)
    
    numOfTrainErrors = np.array(numOfTrainErrors)
    numOfTestErrors = np.array(numOfTestErrors)
    
    print
    print 'Errors in training data sets (10-fold)'
    for i in range(0, len(numOfTrainErrors)):
        print 'Set ' + str(i + 1) + ': ' + str(numOfTrainErrors[i])
    
    print 
    print 'Std deviation of train data: ' + str(numOfTrainErrors.std())
    print 'Mean of training data: ' + str(numOfTrainErrors.mean())   
    
    print
    print 'Errors in test data sets (10-fold)'
    for i in range(0, len(numOfTestErrors)):
        print 'Set ' + str(i + 1) + ': ' + str(numOfTestErrors[i])
         
    print
    print 'Std deviation of test data: ' + str(numOfTestErrors.std())     
    print 'Mean of test data: ' +str(numOfTestErrors.mean())
    
    createBoxPlot(numOfTrainErrors, numOfTestErrors)

    
def calculateErrorCount(clazz,predictedValues):
    return len(filter(None,[predictedValuese if clazze != predictedValuese else '' for clazze, predictedValuese in zip(clazz,predictedValues)]))
 


def inferErrors(xTrain, yTrain, xData, yData):
    regressionModel = LogisticRegression()
    regressionModel = regressionModel.fit(xTrain,yTrain);
    predictedY = regressionModel.predict(xData)
    return calculateErrorCount(yData, predictedY)


def createBoxPlot(trainErrCount, testErrCount):
    data = [trainErrCount, testErrCount]
    colors = ['cyan','pink']
    pp.figure()
    box = pp.boxplot(data, labels=['Training','Test'], patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    pp.show()


def generateLabels():
    return str(labels[0]) + '~' + "+".join(labels[1:])

      
linearRegression(x) # Task 1a)