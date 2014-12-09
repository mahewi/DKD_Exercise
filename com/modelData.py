import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

from sklearn.linear_model import LogisticRegression
from patsy.highlevel import dmatrices

if __name__ == '__main__':
    pass

# Set pointer to correct destination
basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "wine.data"))

# Parse 'wine.data' file into two variables x, y -> (x = labels, y = feature data).
x = np.loadtxt(filepath, delimiter=',')
y = np.loadtxt(filepath, delimiter=',', usecols=range(1, 14))

# Label each feature for sake of clarity. Used in creating plot views.
labels = ['Class','Alcohol', 'Malic acid', 'Ash', 'Alcality of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

def linearRegression(matrix):
    dta = sm.datasets.fair.load_pandas().data
    #print dta
    dta['affair'] = (dta.affairs > 0).astype(int)
    df = pd.DataFrame(matrix,columns=labels)
    #print df
    y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
                  religious + educ + C(occupation) + C(occupation_husb)',
                  dta, return_type="dataframe")
    print y
    regModel = LogisticRegression()
    
linearRegression(x)