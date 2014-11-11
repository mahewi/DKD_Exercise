import numpy as np
from matplotlib import pyplot as pp
import os


if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "wine.data"))

x = np.loadtxt(filepath, delimiter=',', usecols=[0])
y = np.loadtxt(filepath, delimiter=',', usecols=range(1, 14))

features = ['Alcohol', 'Malic acid', 'Ash', 'Alcality of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


for i in range(0, 12):
    hist = []
    for j in range (0, len(y) - 1):
        hist.append(y[j,i])
    pp.grid(True)
    pp.title('Feature ' + `i + 1` + ': ' + features[i])
    pp.ylabel('Count')
    pp.xlabel('Value')
    pp.hist(hist)
    pp.show()
        


'''print x
print y'''


'''pp.title('TITLE')
pp.ylabel('Y-AXIS')
pp.xlabel('X-AXIS')
pp.grid(True)
line = pp.plot(x)
pp.setp(line, color='r', linewidth=4.0, linestyle='--')
pp.show()'''