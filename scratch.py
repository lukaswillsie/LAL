import pickle

import matplotlib.pyplot as plt

accuracies = pickle.load(open("accuracies.pkl", "rb"))
plt.plot(accuracies)
plt.show()