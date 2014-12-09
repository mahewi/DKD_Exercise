import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from matplotlib import pyplot as pp
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from patsy.highlevel import dmatrices
from sklearn.cross_validation import StratifiedKFold as SKF
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
    y, X = dmatrices('Class ~ Alcohol + Malic_acid + Ash + Alcality_of_ash'+
                     '+ Magnesium + Total_phenols + Flavanoids + Nonflavanoid_phenols + Proanthocyanins + Color_intensity'+
                     '+ Hue + OD280_OD315_of_diluted_wines + Proline', 
                     df, return_type='matrix')
    y = np.ravel(y)
    regModel = LogisticRegression()
    regModel = regModel.fit(X,y);
    predictedY = regModel.predict(X)
    numOfErrors =  calculateNumOfErrors(y, predictedY)
    print 'Number of errors in whole data: \n' + str(numOfErrors)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=np.random)
    regModelTrain = LogisticRegression()
    regModelTrain = regModelTrain.fit(X_train,y_train)
    predictedYTrain = regModelTrain.predict(X_train)
    numOfErrorsTrain = calculateNumOfErrors(y_train, predictedYTrain)
    print 'Number of errors in training data 2: \n' + str(numOfErrorsTrain)
    
    regModelTest = LogisticRegression()
    regModelTest = regModelTest.fit(X_test,y_test)
    predictedYTest = regModelTest.predict(X_test)
    numOfErrorsTest = calculateNumOfErrors(y_test, predictedYTest)
    print 'Number of errors in test data 2: \n' + str(numOfErrorsTest)
    sss = SSS(y,10,test_size=0.25,random_state=np.random)
    i = 0
    numOfTrainErrors = []
    numOfTestErrors = []
    for train_index, test_index in sss:
        i = i+1
        #print('TRAIN:', train_index, 'TEST:', test_index)
        X_2train, X_2test = X[train_index],X[test_index]
        y_2train,y_2test = y[train_index],y[test_index]
        regModelTrain = LogisticRegression()
        regModelTrain = regModelTrain.fit(X_2train, y_2train)
        predictedY = regModelTrain.predict(X_2train)
        numOfErrorsTrain = calculateNumOfErrors(y_2train, predictedY)
        numOfTrainErrors.append(numOfErrorsTrain)
        print 'Number of errors in training data (10-fold) ' + str(i) + ':\n' + str(numOfErrorsTrain)
        regModelTest = LogisticRegression()
        regModelTest = regModelTest.fit(X_2test,y_2test)
        predictedYTest = regModelTest.predict(X_2test)
        numOfErrorsTest = calculateNumOfErrors(y_2test, predictedYTest)
        numOfTestErrors.append(numOfErrorsTest)
        print 'Number of errors in test data (10-fold) ' + str(i) + ':\n' + str(numOfErrorsTest)
        
    numOfTrainErrors = np.array(numOfTrainErrors)
    numOfTestErrors = np.array(numOfTestErrors)
    print 'Std deviation of train data: ' + str(numOfTrainErrors.std())
    print 'Std deviation of test data: ' + str(numOfTestErrors.std())
    print 'Mean of train data: ' + str(numOfTrainErrors.mean())    
    print 'Mean of test data: ' +str(numOfTestErrors.mean())
    
    data = [numOfTrainErrors, numOfTestErrors]
    colors = ['cyan','pink']
    pp.figure()
    box = pp.boxplot(data, labels=['Train','Test'], patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    pp.show()
    
def calculateNumOfErrors(clazz,predictedValues):
    return len(filter(None,[predictedValuese if clazze != predictedValuese else '' for clazze, predictedValuese in zip(clazz,predictedValues)]))
      
linearRegression(x)