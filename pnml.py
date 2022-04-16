import copy
import pickle

import numpy as np
import torch
from Classes.dataset import DatasetMNIST
from Classes.models import SimpleMLP
from scipy import stats
from torch import optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def evaluate(m):
    with torch.no_grad():
        pred = m(dataset.testData)
        if is_binary:
            pred = torch.sigmoid(pred)
            pred = (pred >= 0.5).long()
        else:
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1, keepdim=True)
        return (torch.sum(pred == dataset.testLabels.long()) / pred.shape[0]).item()


def predict_probabilities(m, inputs):
    pred = m(inputs)
    if is_binary:
        pos_prob = torch.sigmoid(pred)
        return torch.cat((1 - pos_prob, pos_prob), dim=1)
    else:
        return torch.softmax(pred, dim=1)


multiclass_loss = CrossEntropyLoss()


def multiclass_criterion(outputs, labels):
    # CrossEntropyLoss requires flattened labels
    return multiclass_loss(outputs, labels.view(-1))


def fit(target_model, inputs, labels, epochs, lr, early_stopping_patience=None):
    criterion = BCEWithLogitsLoss() if is_binary else multiclass_criterion
    optimizer = optim.Adam(
        target_model.parameters(), lr=lr
    )
    best_loss = float("inf")
    early_stop = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        train_loss = criterion(target_model(inputs), labels)
        # print(train_loss)

        train_loss.backward()
        optimizer.step()

        if early_stopping_patience:
            if train_loss.item() < best_loss:
                early_stop = 0
                best_loss = train_loss.item()
            else:
                early_stop += 1

            if early_stop > early_stopping_patience:
                return

    return


def selectNext():
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
                    (torch.Tensor([[t]]) if is_binary else torch.LongTensor([[t]])).cuda()
                ),
                dim=0
            )
            fit(temp_model, train_data, train_labels, 1000, 1e-2, early_stopping_patience=30)
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
dataset = DatasetMNIST()
dataset.setStartState(10)

dataset.trainData = torch.from_numpy(dataset.trainData).float().cuda()
dataset.trainLabels = torch.from_numpy(dataset.trainLabels).float().cuda()
dataset.testData = torch.from_numpy(dataset.testData).float().cuda()
dataset.testLabels = torch.from_numpy(dataset.testLabels).float().cuda()

classes = torch.unique(dataset.trainLabels)
is_binary = len(classes) == 2

if not is_binary:
    dataset.trainLabels = dataset.trainLabels.long()
    classes = torch.unique(dataset.trainLabels)

# model = SimpleMLP([2,10,10,1])
model = SimpleMLP([784, 10])
model.cuda()

accuracies = []
for experiment in range(experiments):
    for iteration in range(iterations):
        # 1. Train the model
        # 2. Evaluate the model
        # 3. Select the next point
        known_data = dataset.trainData[dataset.indicesKnown, :]
        known_labels = dataset.trainLabels[dataset.indicesKnown, :]
        fit(model, known_data, known_labels, 1000, 1e-2, early_stopping_patience=40)
        accuracies.append(evaluate(model))
        print(accuracies[-1])
        pickle.dump(accuracies, open("accuracies.pkl", "wb"))
        selectNext()
