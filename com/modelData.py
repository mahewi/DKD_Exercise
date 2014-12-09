import numpy as np
import os

from sklearn.linear_model import LogisticRegression

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

def linearRegression(matrix):
    regModel = LogisticRegression()