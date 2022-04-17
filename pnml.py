import copy
import time

import numpy as np
import torch
from Classes.dataset import DatasetMNIST, DatasetCheckerboard2x2
from Classes.models import SimpleMLP
from scipy import stats
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from util import fit, predict_probabilities, Metrics, init_model


def select_next_pnml():
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
                    (torch.Tensor([[t]]) if dataset.is_binary else torch.LongTensor([[t]]))
                ),
                dim=0
            )
            fit(*fit_params, lambda m: loss_function(m(train_data), train_labels),
                early_stopping_patience=40)  # Get the probability predicted for class t
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
        if points_seen % 100 == 0:
            print(points_seen)

    dataset.indicesKnown = torch.cat((dataset.indicesKnown, torch.LongTensor([selectedIndex])))
    dataset.indicesUnknown = torch.cat((dataset.indicesUnknown[:selectedIndex1toN], dataset.indicesUnknown[selectedIndex1toN+1:]))


def select_next_random():
    selected = torch.randint(low=0, high=dataset.indicesUnknown.shape[0], size=(1,)).item()
    dataset.indicesKnown = torch.cat((dataset.indicesKnown, torch.LongTensor([selected])))
    dataset.indicesUnknown = torch.cat((dataset.indicesUnknown[:selected], dataset.indicesUnknown[selected+1:]))


def select_next_uncertainty():
    unknown_prob = predict_probabilities(model, dataset.trainData[dataset.indicesUnknown, :]).numpy()
    uncertainty = stats.entropy(unknown_prob, axis=1)
    most_uncertain_index = np.argmax(uncertainty, axis=0)
    dataset.indicesKnown = torch.cat(
        (dataset.indicesKnown, torch.LongTensor([dataset.indicesUnknown[most_uncertain_index]]))
    )
    dataset.indicesUnknown = torch.cat(
        (dataset.indicesUnknown[:most_uncertain_index], dataset.indicesUnknown[most_uncertain_index+1:])
    )


experiments = 5
iterations = 200
method = "uncertainty"

select_next_options = {
    "rand": select_next_random,
    "pnml": select_next_pnml,
    "uncertainty": select_next_uncertainty
}

dataset = DatasetMNIST(seed=42)
dataset.set_is_binary()

dataset.trainData = torch.from_numpy(dataset.trainData).float()
dataset.trainLabels = torch.from_numpy(dataset.trainLabels).float()
dataset.testData = torch.from_numpy(dataset.testData).float()
dataset.testLabels = torch.from_numpy(dataset.testLabels).float()

# CrossEntropyLoss(), which we use for multiclass classification, requires long labels
if not dataset.is_binary:
    dataset.trainLabels = dataset.trainLabels.long()
    dataset.testLabels = dataset.testLabels.long()

classes = torch.unique(dataset.trainLabels)

model = None
fit_params = None
if isinstance(dataset, DatasetCheckerboard2x2):
    # model = SimpleMLP([2, 5, 10, 5, 1])
    # fit_params = (model, 100, 1e-2)
    model = SimpleMLP([2, 10, 10, 1])
    fit_params = (model, 100, 1e-2)
elif isinstance(dataset, DatasetMNIST):
    model = SimpleMLP([784, 10])
    fit_params = (model, 1000, 1e-2)

if not model:
    print("ERROR: Haven't implemented this script for the chosen dataset")
    exit(-1)

multiclass_loss = CrossEntropyLoss()


def multiclass_criterion(outputs, labels):
    # CrossEntropyLoss requires flattened labels
    return multiclass_loss(outputs, labels.view(-1))


loss_function = BCEWithLogitsLoss() if dataset.is_binary else multiclass_criterion


name = method + "-" + ("mnist" if isinstance(dataset, DatasetMNIST) else "checkerboard2x2")
metrics = Metrics(name, method)

select_next = select_next_options[method]
for experiment in range(experiments):
    # Reset the dataset
    dataset.set_start_state_torch(len(classes))
    # Start tracking fresh data
    metrics.new_experiment()
    # Randomly initialize the model weights
    model.apply(init_model)
    print(f"Experiment {experiment + 1}", end="")
    for iteration in range(iterations):
        start = time.time()
        # Train the model on the points that have been chosen to be labelled
        known_data = dataset.trainData[dataset.indicesKnown, :]
        known_labels = dataset.trainLabels[dataset.indicesKnown, :]
        fit(*fit_params, lambda m: loss_function(m(known_data), known_labels),
            early_stopping_patience=40)

        # Evaluate the model
        metrics.evaluate(model, dataset, loss_function)

        # Select the next point to be labelled
        select_next()
        print(".", end="")
        end = time.time()
        print(end - start)
    print()
    metrics.save()

metrics.plot()
