import matplotlib.pyplot as plt
import numpy as np

pos_fraction = np.linspace(0.00, 1.00, 1000)
pos_fraction[:10]
gini = 1 - pos_fraction**2 - (1-pos_fraction)**2

plt.plot(pos_fraction, gini)
plt.ylim(0, 1)
plt.xlabel('Positive fraction')
plt.ylabel('Gini impurity')
plt.show()

def gini_impurity(label):
    # When the set is empty it is pure
    if len(label) == 0:
        return 0
    # Count occurence of each label
    count = np.unique(label, return_counts=True)[1]
    fractions = count / float(len(label))
    return 1 - np.sum(fractions**2)

labels = [
    [1, 1, 0, 1],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0]
]

p = lambda x: print(x)

list(map(p, list(map(gini_impurity, labels))))
