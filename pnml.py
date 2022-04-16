import copy
import pickle
import time

import numpy as np
import torch
from Classes.dataset import DatasetMNIST, DatasetCheckerboard2x2
from Classes.models import SimpleMLP
from scipy import stats
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from util import fit, evaluate, predict_probabilities


def select_next():
    # 1. For each unlabelled point x
    #   i. For each label t
    #       a. Add the (x, t) pair to the data and fit
    #       b. Get p(t|x) and store
    #   ii. Normalize p(t | x) for all t
    #   iii. Compute uncertainty using some metric
    #   iv. Update index of most uncertain point if necessary
    max_uncertainity = float("-inf")
    selectedIndex = -1
    selectedIndex1toN = -1

    known_data = dataset.trainData[dataset.indicesKnown, :]
    known_labels = dataset.trainLabels[dataset.indicesKnown, :]
    points_seen = 0
    for i, unknown_index in enumerate(dataset.indicesUnknown):
        label_probabilities = []
        for j, t in enumerate(classes):
            temp_model = copy.deepcopy(model)
            train_data = torch.cat(
                (
                    known_data,
                    dataset.trainData[(unknown_index,), :]
                ),
                dim=0
            )
            train_labels = torch.cat(
                (
                    known_labels,
                    (torch.Tensor([[t]]) if dataset.is_binary else torch.LongTensor([[t]]))#.cuda()
                ),
                dim=0
            )
            fit(temp_model, 1000, 1e-2, lambda m: loss_function(m(train_data), train_labels), early_stopping_patience=30)
            # Get the probability predicted for class t
            pred = predict_probabilities(temp_model, dataset.trainData[(unknown_index,), :])[:, j]
            label_probabilities.append(pred.item())
        label_probabilities = np.array(label_probabilities)
        # stats.entropy() will automatically normalize
        entropy = stats.entropy(label_probabilities)
        if entropy > max_uncertainity:
            max_uncertainity = entropy
            selectedIndex = unknown_index
            selectedIndex1toN = i
        points_seen += 1
        print(points_seen)

    dataset.indicesKnown = np.concatenate(([dataset.indicesKnown, np.array([selectedIndex])]))
    dataset.indicesUnknown = np.delete(dataset.indicesUnknown, selectedIndex1toN)


experiments = 1
iterations = 100
dataset = DatasetCheckerboard2x2()
dataset.setStartState(2)
dataset.set_is_binary()

dataset.trainData = torch.from_numpy(dataset.trainData).float()
dataset.trainLabels = torch.from_numpy(dataset.trainLabels).float()
dataset.testData = torch.from_numpy(dataset.testData).float()
dataset.testLabels = torch.from_numpy(dataset.testLabels).float()


if not dataset.is_binary:
    dataset.trainLabels = dataset.trainLabels.long()

classes = torch.unique(dataset.trainLabels)

model = SimpleMLP([2,10,10,1])
# model = SimpleMLP([784, 10])
# model.cuda()

multiclass_loss = CrossEntropyLoss()


def multiclass_criterion(outputs, labels):
    # CrossEntropyLoss requires flattened labels
    return multiclass_loss(outputs, labels.view(-1))


loss_function = BCEWithLogitsLoss() if dataset.is_binary else multiclass_criterion


accuracies = []
for experiment in range(experiments):
    for iteration in range(iterations):
        # 1. Train the model
        # 2. Evaluate the model
        # 3. Select the next point
        known_data = dataset.trainData[dataset.indicesKnown, :]
        known_labels = dataset.trainLabels[dataset.indicesKnown, :]
        fit(model, 1000, 1e-2, lambda m: loss_function(m(known_data), known_labels), early_stopping_patience=40)
        accuracies.append(evaluate(model, dataset))
        print(accuracies[-1])
        pickle.dump(accuracies, open("accuracies.pkl", "wb"))
        select_next()
