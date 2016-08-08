"""

Train perceptron using tutorial perceptron class


"""


import perceptronClass
import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.evaluate import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# setosa and versicolor
y = df.iloc[0:100, 4].values
y = perceptronClass.np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length
X = df.iloc[0:100, [0,2]].values




ppn = perceptronClass.Perceptron(learningRate = 0.1, iterations = 10)

ppn.Train(X, y)
print('Weights: %s' % ppn.weight)

plot_decision_regions(X, y, clf = ppn)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors)+1), ppn.errors, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.show()

