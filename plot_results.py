import pickle

import matplotlib.pyplot as plt
import numpy as np
from Classes.dataset import DatasetCheckerboard2x2, DatasetCheckerboard4x4, Dataset, DatasetRotatedCheckerboard2x2


def get_metrics_objs(dataset):
    metrics = []
    try:
        rand = pickle.load(open(f"metrics/rand-{dataset}.pkl", "rb"))
        rand.method = "RS"
        metrics.append(rand)
    except FileNotFoundError:
        pass

    try:
        pnml = pickle.load(open(f"metrics/pnml-{dataset}.pkl", "rb"))
        pnml.method = "pNMLAL"
        metrics.append(pnml)
    except FileNotFoundError:
        pass

    try:
        uncertainty = pickle.load(open(f"metrics/uncertainty-{dataset}.pkl", "rb"))
        uncertainty.method = "US"
        metrics.append(uncertainty)
    except FileNotFoundError:
        pass

    try:
        apnml = pickle.load(open(f"metrics/apnml-{dataset}.pkl", "rb"))
        apnml.method = "apNMLAL"
        metrics.append(apnml)
    except FileNotFoundError:
        pass

    return tuple(metrics)


colour_map = {
    "RS": "#FA2E2E",
    "pNMLAL": "#FFA73E",
    "US": "#39E04D",
    "apNMLAL": "#338CFA"
}


fig, axs = plt.subplots(2, 2)


def make_experiment_plot(*metrics, axis, dataset_name: str = "", dataset: Dataset = None):
    # axis.set_title(dataset_name)
    methods = []
    for metric_obj in metrics:
        methods.append(metric_obj.method)
        y = np.mean(np.array(metric_obj.validation_accuracy), axis=0)
        x = np.arange(y.shape[0]) + 2
        axis.plot(x, y, label=metric_obj.method, color=colour_map[metric_obj.method])
        # axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    # plt.legend(loc='lower right')
    # Put a small version of the dataset in the bottom-right of the plot
    if dataset:
        a = axis.inset_axes([0.67, 0.10, 0.25, 0.25])
        mask = (dataset.trainLabels == 1).reshape(-1)
        a.spines['top'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xticks([])
        a.set_yticks([])
        a.scatter(dataset.trainData[mask, :][:, 0], dataset.trainData[mask, :][:, 1], color="red", marker=".", alpha=0.7)
        a.scatter(dataset.trainData[~mask, :][:, 0], dataset.trainData[~mask, :][:, 1], color="blue", marker=".", alpha=0.7)


def get_subset_of_dataset(num_points_per_class, dataset):
    mask = (dataset.trainLabels == 1).reshape(-1)
    pos_points = np.where(mask)[0]
    neg_points = np.where(~mask)[0]

    pos_select = np.random.choice(range(pos_points.shape[0]), size=num_points_per_class, replace=False)
    neg_select = np.random.choice(range(neg_points.shape[0]), size=num_points_per_class, replace=False)

    pos_points = pos_points[pos_select]
    neg_points = neg_points[neg_select]
    points = np.concatenate((pos_points, neg_points))

    dataset.trainData = dataset.trainData[points]
    dataset.trainLabels = dataset.trainLabels[points]


checkerboard_2x2 = DatasetCheckerboard2x2()
get_subset_of_dataset(50, checkerboard_2x2)

checkerboard4x4 = DatasetCheckerboard4x4()
get_subset_of_dataset(100, checkerboard4x4)

rotated_checkerboard2x2 = DatasetRotatedCheckerboard2x2()
get_subset_of_dataset(50, rotated_checkerboard2x2)

make_experiment_plot(*get_metrics_objs("rotated-checkerboard2x2"), dataset_name="Rotated Checkerboard2x2", dataset=rotated_checkerboard2x2, axis=axs[1, 0])
# make_experiment_plot(*get_metrics_objs("mnist"), dataset_name="MNIST", dataset=None, axis=axs[1, 1])
make_experiment_plot(*get_metrics_objs("checkerboard2x2"), dataset_name="Checkerboard2x2", dataset=checkerboard_2x2, axis=axs[0, 0])
make_experiment_plot(*get_metrics_objs("checkerboard4x4"), dataset_name="Checkerboard4x4", dataset=checkerboard4x4, axis=axs[1, 1])
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['left'].set_visible(False)
axs[0,1].spines['bottom'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
handles, labels = axs[0,0].get_legend_handles_labels()
leg = fig.legend(handles, labels, bbox_to_anchor=(0.75, .75), loc='center')
leg.get_frame().set_linewidth(0.0)
plt.savefig(f"./new-out/validation_accuracies.png", bbox_inches="tight")
plt.show()
