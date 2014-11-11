import numpy as np
from matplotlib import pyplot as pp
import os
from numpy.f2py.auxfuncs import throw_error


if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "wine.data"))

x = np.loadtxt(filepath, delimiter=',', usecols=[0])
y = np.loadtxt(filepath, delimiter=',', usecols=range(1, 14))

features = ['Alcohol', 'Malic acid', 'Ash', 'Alcality of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

        
def defaultHistogram(matrix):
    for i in range(0, 13):
        hist = []
        for j in range (0, len(y)):
            hist.append(matrix[j,i])
        pp.grid(True)
        pp.title('Feature ' + `i + 1` + ': ' + features[i])
        pp.ylabel('Count')
        pp.xlabel('Value')
        pp.hist(hist, 30, facecolor='#800020', normed=1, alpha=0.8)
        pp.show()


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
                throw_error
        pp.grid(True)
        pp.title('Feature ' + `i + 1` + ': ' + features[i])
        pp.ylabel('Count')
        pp.xlabel('Value')
        pp.hist([hist1, hist2, hist3], 15, color=['#800020', '#F7E7CE', '#673147'], label=['Class1', 'Class2', 'Class3'],
                 normed=1, alpha=0.8)
        pp.legend()
        pp.show()
        
# One of the features: 'Proline', has significantly greater values than the other features. Therefore it dominates
# the plot        
def parallelFeatures(matrix):
    for i in range(0, 13):
        plot = []
        for j in range (0, len(y)):
            plot.append(matrix[j,i])
        pp.grid(True)
        pp.ylabel('Value')
        pp.xlabel('Count')
        pp.plot(plot, color='#800020', alpha=0.8)
    pp.title('Parallel plot of features')
    pp.show()
    
    
def calculateCorrelations(matrix):
    plotMatrix = []
    for i in range(0, 13):
        plot = []
        for j in range(0, len(y)):
            plot.append(matrix[j][i])
        plotMatrix.append(plot)

    for k in range(0, len(plotMatrix)):
        for z in range(k + 1, len(plotMatrix)):
            array = np.corrcoef(plotMatrix[k], plotMatrix[z])
            for r in range(0, len(array)):
                if 0.7 <= array[r,1] <= 1 or -0.7 >= array[r,1] <= -1:
                    print 'FeatureA: '
                    print k
                    print 'FeatureB: '
                    print z
                    print 'correlation factor: ' 
                    print array[r,1]
                break
                    
#defaultHistogram(y)
#visualizeByClasses(x, y)
#parallelFeatures(y)
calculateCorrelations(y)
