import numpy as np
from matplotlib import pyplot as pp
import os


if __name__ == '__main__':
    pass

basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, "..", "wine.data"))

x = np.loadtxt(filepath, delimiter=',', usecols=[0])
y = np.loadtxt(filepath, delimiter=',', skiprows = 0)[1:]

print x
print y


'''pp.title('TITLE')
pp.ylabel('Y-AXIS')
pp.xlabel('X-AXIS')
pp.grid(True)
line = pp.plot(x)
pp.setp(line, color='r', linewidth=4.0, linestyle='--')
pp.show()'''