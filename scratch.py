import pickle

import matplotlib.pyplot as plt

metric = pickle.load(open("metrics/apmnl unbalanced.pkl", "rb"))

metric.plot()

