import numpy as np
import matplotlib
from matplotlib import pyplot as pp
import os
from sklearn.preprocessing.data import StandardScaler


if __name__ == '__main__':
    pass


# Set pointer to correct destination
basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "wine.data"))

# Parse 'wine.data' file into two variables x, y -> (x = labels, y = feature data).
x = np.loadtxt(filepath, delimiter=',', usecols=[0])
y = np.loadtxt(filepath, delimiter=',', usecols=range(1, 14))

# Label each feature for sake of clarity. Used in creating plot views.
featureLabels = ['Alcohol', 'Malic acid', 'Ash', 'Alcality of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


# Task 1: Plot a histogram of each feature       
def defaultHistogram(matrix):
    for i in range(0, 13):
        hist = []
        for j in range (0, len(y)):
            hist.append(matrix[j,i])
        pp.grid(True)
        pp.title('Feature ' + `i + 1` + ': ' + featureLabels[i])
        pp.ylabel('Count')
        pp.xlabel('Value')
        pp.hist(hist, 30, facecolor='#800020', normed=1, alpha=0.8)
        pp.show()


# Task 2: Plot the same histograms and visualize the classes. 
def visualizeByClasses(array, matrix):
    for i in range(0, 13):
        hist1 = []
        hist2 = []
        hist3 = []
        for j in range (0, len(y)):
            if array[j] == 1:
                hist1.append(matrix[j,i])
            elif array[j] == 2:
                hist2.append(matrix[j,i])
            elif array[j] == 3:
                hist3.append(matrix[j,i])
            else:
                return
        pp.grid(True)
        pp.title('Feature ' + `i + 1` + ': ' + featureLabels[i])
        pp.ylabel('Count')
        pp.xlabel('Value')
        pp.hist(hist1, 15, color="r", label="Class1", alpha=0.8, normed=1)
        pp.hist(hist2, 15, color="b", label="Class2", alpha=0.8, normed=1)
        pp.hist(hist3, 15, color="g", label="Class3", alpha=0.8, normed=1)
        pp.legend()
        pp.show()
        

# Task 3: Make a parellel plot of the features. 
# One of the features: 'Proline', has/maps to significantly larger values than the other features. 
# Therefore it dominates the plot over other features       
def parallelFeatures(matrix):
    red_patch = matplotlib.patches.Patch(color='r', label='Class1')
    green_patch = matplotlib.patches.Patch(color='g', label='Class2')
    blue_patch = matplotlib.patches.Patch(color='b', label='Class3')
    colors = {
             1: 'r',
             2: 'g',
             3: 'b'
             }
    for i in range(0, len(y)):
        sample = []
        for j in range (0, 13):
            sample.append(matrix[i,j])
        pp.grid(True)
        pp.ylabel('Value')
        pp.xlabel('Count')
        pp.plot(sample, color=colors.get(x[i]), alpha=0.8)
    pp.title('Parallel plot of features')
    pp.legend(handles=[red_patch,green_patch,blue_patch])
    pp.show()
    

# Task 4: Calculate the correlation coefficients.
# The correlation coeffecients are saved to the 'correlations' variable (matrix).
# The strongly correlating features (between [0,7..1 or -0,7..-1]) are printed also    
def calculateCorrelations(matrix):
    featureMatrix = initializeFeatureMatrix()
    interestingCorrelations = []

    for k in range(0, len(featureMatrix)):
        for z in range(k + 1, len(featureMatrix)):
            correlations = np.corrcoef(featureMatrix[k], featureMatrix[z])
            print correlations 
            for r in range(0, len(correlations)):
                if 0.7 <= correlations[r,1] <= 1 or -0.7 >= correlations[r,1] <= -1:
                    interestingCorrelations.append(k)
                    interestingCorrelations.append(z)
                    print
                    print 'FeatureA: ' + str(featureLabels[k]) + '(' + str(k) + ')'
                    print 'FeatureB: ' + str(featureLabels[z]) + '(' + str(z) + ')'
                    print 'Correlation factor: ' + str(correlations[r,1])
                    print 
                break
    return interestingCorrelations


# Task 5: Visualize the interesting correlations with the scatter plot
def visualizeAsScatterPlot(matrix):
    strongCorrelations = calculateCorrelations(y)
    
    i = 0
    while i < len(strongCorrelations):
        first = strongCorrelations[i]
        second = strongCorrelations[i + 1]
        i = i + 2
        firstFeatureData = matrix[:,first]
        secondFeatureData = matrix[:, second]
        
        pp.title('Scatter plot of features: ' + featureLabels[first] + ' & ' + featureLabels[second])
        pp.scatter(firstFeatureData, secondFeatureData, s=60, c=['red', 'blue'], label='Scatter', alpha=0.5)
        pp.legend()
        pp.show()    


# Task 6: Calculate Covariance matrix -> calculate eigenvalues and eigenvectors from square covariance matrix
def calculateConvarianceAndEigen(matrix):
    featureMatrix = initializeFeatureMatrix()
    # featureCovariance = np.cov(featureData) # From each feature -- 1-D array
    # covarianceMatrix =  np.cov(tempMatrix) # From each feature matrix -- 2-D array
    bigCovarianceMatrix = np.cov(featureMatrix) # From the whole data set matrix -- 2-D array
    eigVal, eigVec = np.linalg.eig(bigCovarianceMatrix)
    print
    print 'Eigenvalues: ' + str(eigVal)
    print
    print 'Eigenvectors: ' + str(eigVec)
    
    return eigVal, eigVec
    

# Task 7: Calculate PCA and generate scatter plot for first two principal components
def calculatePrincipalProjection(matrix):
    featureMatrix = initializeFeatureMatrix()
    eigVal, eigVec = calculateConvarianceAndEigen(matrix)

    eigPairs = [(np.abs(eigVal[i]), eigVec[:,i]) for i in range(len(eigVal))]
    eigPairs.sort(reverse=True)
    '''for i in eigPairs:
        print (i[0])'''
    
    dimMatrix = np.hstack(eigPairs[0][1])
    pcaResult = dimMatrix.T.dot(featureMatrix)
    
    #pcaResult = matplotlib.mlab.PCA(featureMatrix.T)
    
    pp.scatter(pcaResult[0:89],pcaResult[89:178], c='red', label='Set 1')
    pp.scatter(pcaResult[0:89], pcaResult[89:178], c='blue', label='Set 2')

    pp.legend()
    pp.show()
       

# Initialize matrix containing arrays of features    
def initializeFeatureMatrix():
    tempMatrix = []
    for i in range(0, 13):
        featureMatrix = []
        for j in range (0, len(y)):
            featureMatrix.append(y[j,i])
        tempMatrix.append(featureMatrix)
    npArr = np.array(tempMatrix)
    return npArr


# The primary method calls are located here.                    
# defaultHistogram(y) # Task 1
# visualizeByClasses(x, y) # Task 2
# parallelFeatures(y) # Task 3
# calculateCorrelations(y) # Task 4
# visualizeAsScatterPlot(y) # Task 5
# calculateConvarianceAndEigen(y) # Task 6
# calculatePrincipalProjection(y) # Task 7

