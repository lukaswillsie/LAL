import pickle

import matplotlib.pyplot as plt
from util import plot_combined_metrics

metric_apnml = pickle.load(open("metrics/apmnl_2x2_acc_SVI.pkl", "rb"))
metric_rand = pickle.load(open("metrics/rand-checkerboard2x2.pkl", "rb"))
metric_uncertainty = pickle.load(open("metrics/uncertainty-checkerboard2x2.pkl", "rb"))

plot_combined_metrics(metric_apnml, metric_rand, metric_uncertainty)

