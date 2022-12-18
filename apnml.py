import copy
import time

import numpy as np
import torch
from Classes.dataset import DatasetMNIST, DatasetCheckerboard2x2, DatasetCheckerboard4x4, DatasetRotatedCheckerboard2x2, DatasetSimulatedUnbalanced
from Classes.models import SimpleMLP
from scipy import stats
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from Classes.svi import GaussianSVI
from util import fit, Metrics, predict_probabilities, init_model, predict_probabilities_with_grad


def log_prior(latent):
    normal = torch.distributions.normal.Normal(0, 1)
    return torch.sum(normal.log_prob(latent), axis=-1).to(device)


def log_likelihood(latent):
    batch_size = latent.shape[0]
    result = torch.zeros(batch_size, requires_grad=True)
    result = result.to(device)
    # print(f"!!!!!!!!!!!!{result.shape}")
    known_data = dataset.trainData[dataset.indicesKnown, :]
    known_labels = dataset.trainLabels[dataset.indicesKnown, :]
    known_data = known_data.to(device)
    known_labels = known_labels.to(device)
    for n in range(batch_size):
        model.set_parameters(latent[n, :])
        model.to(device)
        probabilities = predict_probabilities_with_grad(model, known_data).to(device)
        log_prob = torch.sum(torch.log(torch.maximum(torch.gather(probabilities, dim=1, index=known_labels.long().to(device)), torch.Tensor([1e-9]).to(device))))
        if n == 0:
            # print("RESULT")
            # print(log_prob.shape)
            result = log_prob.unsqueeze(dim=0)
        else:
            # print(result.shape)
            result = torch.cat((result, log_prob.unsqueeze(dim=0)), dim=0)
        # result[n] = log_prob
    result = result.to(device)
    return result


def log_joint(latent):
    return log_likelihood(latent) + log_prior(latent)


def get_approximate_posterior():
    # Hyperparameters
    n_iters = 1000
    num_samples_per_iter = 75

    svi = GaussianSVI(true_posterior=log_joint, num_samples_per_iter=num_samples_per_iter)

    # Set up optimizer.
    D = model.total_params
    init_mean = torch.randn(D)
    init_mean.requires_grad = True
    init_log_std  = torch.randn(D)
    init_log_std.requires_grad = True
    init_params = (init_mean, init_log_std)

    params = init_params

    def callback(params, t):
        if t % 100 == 0:
            print("Iteration {} lower bound {}".format(t, svi.objective(params)))

    def update(params):
        optim = torch.optim.SGD(params, lr=1e-5, momentum=0.9)
        optim.zero_grad()
        loss = svi.objective(params)
        loss.backward()
        optim.step()
        return params

    # Main loop.
    print("Optimizing variational parameters...")
    for i in range(n_iters):
        params = update(params)
        callback(params, i)

    params[0].requires_grad = False
    params[1].requires_grad = False

    return params


def criterion(m, inputs, labels, svi_mean, svi_logstd, multiclass=False):
    probability = predict_probabilities_with_grad(m, inputs)
    # probability.requires_grad = True
    # print(f"probability: {probability}")
    # print(f"len of m.get_parameters: {len(m.get_parameters())}")
    # print(f"{torch.log(torch.Tensor([0]))}")
    probability = probability.to(device)
    eps=1e-7
    return -(torch.sum(torch.log(torch.gather(probability, dim=1, index=labels.long()) + eps)) + GaussianSVI.diag_gaussian_logpdf(m.get_parameters(), svi_mean, svi_logstd))


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

    (svi_mean, svi_log_std) = get_approximate_posterior()
    svi_mean.requires_grad = False
    svi_log_std.requires_grad = False
    svi_mean = svi_mean.to(device)
    svi_log_std = svi_log_std.to(device)

    # print(f"svi_mean, svi_log_std: {svi_mean}, {svi_log_std}")

    points_seen = 0
    for i, unknown_index in enumerate(dataset.indicesUnknown):
        label_probabilities = []
        for j, t in enumerate(classes):
            temp_model = copy.deepcopy(model)
            temp_model.to(device)
            train_data = torch.cat(
                (
                    known_data,
                    dataset.trainData[(unknown_index,), :]
                ),
                dim=0
            ).to(device)
            train_labels = torch.cat(
                (
                    known_labels,
                    (torch.Tensor([[t]]) if is_binary else torch.LongTensor([[t]])).to(device)
                ),
                dim=0
            ).to(device)
            fit(temp_model, *fit_params, lambda m: criterion(m, train_data, train_labels, svi_mean, svi_log_std, not dataset.is_binary), early_stopping_patience=30, debug=False)
            # Get the probability predicted for class t
            pred = predict_probabilities(temp_model, dataset.trainData[(unknown_index,), :])[:, j]
            pred.requires_grad = False
            label_probabilities.append(pred.item())
            del train_data
            del train_labels
            del temp_model
        label_probabilities = np.array(label_probabilities)
        # print(f"label probabilities: {label_probabilities}")
        # stats.entropy() will automatically normalize
        entropy = stats.entropy(label_probabilities)
        # print(f"entropy of point i: {entropy}")
        if entropy > max_uncertainity:
            max_uncertainity = entropy
            selectedIndex = unknown_index
            selectedIndex1toN = i
        points_seen += 1

    dataset.indicesKnown = np.concatenate(([dataset.indicesKnown, np.array([selectedIndex])]))
    dataset.indicesUnknown = np.delete(dataset.indicesUnknown, selectedIndex1toN)


experiments = 5
iterations = 100
dataset = DatasetCheckerboard4x4(seed=42)

# dataset = DatasetMNIST(seed=42)
# dataset.set_is_binary()

dataset.trainData = torch.from_numpy(dataset.trainData).float()
dataset.trainLabels = torch.from_numpy(dataset.trainLabels).float()
dataset.testData = torch.from_numpy(dataset.testData).float()
dataset.testLabels = torch.from_numpy(dataset.testLabels).float()

classes = torch.unique(dataset.trainLabels)
is_binary = len(classes) == 2

if not is_binary:
    dataset.trainLabels = dataset.trainLabels.long()
    classes = torch.unique(dataset.trainLabels)

# model = SimpleMLP([2,10,10,1])
# model = SimpleMLP([2,10,10,1])
# fit_model = SimpleMLP([2,10,10,1])



multiclass_loss = CrossEntropyLoss()

def multiclass_criterion(outputs, labels):
    # CrossEntropyLoss requires flattened labels
    return multiclass_loss(outputs, labels.view(-1).long())

model = None
fit_params = None
if isinstance(dataset, DatasetCheckerboard2x2) or isinstance(dataset, DatasetRotatedCheckerboard2x2):
    # model = SimpleMLP([2, 5, 10, 5, 1])
    # fit_params = (model, 100, 1e-2)
    model = SimpleMLP([2,10,10,1])
    fit_model = SimpleMLP([2,10,10,1])
    fit_params = (100, 1e-2)
    loss_function = BCEWithLogitsLoss()
elif isinstance(dataset, DatasetSimulatedUnbalanced):
    # model = SimpleMLP([2, 5, 10, 5, 1])
    # fit_params = (model, 100, 1e-2)
    model = SimpleMLP([2, 1])
    fit_model = SimpleMLP([2,10,10,1])
    fit_params = (100, 1e-3)
    loss_function = BCEWithLogitsLoss()
elif isinstance(dataset, DatasetCheckerboard4x4):
    # model = SimpleMLP([2, 5, 10, 5, 1])
    # fit_params = (model, 100, 1e-2)
    model = SimpleMLP([2, 10, 10, 1])
    fit_model = SimpleMLP([2, 10, 10, 1])
    fit_params = (200, 32e-3)
    loss_function = BCEWithLogitsLoss()
elif isinstance(dataset, DatasetMNIST):
    model = SimpleMLP([784, 10])
    fit_model = SimpleMLP([784, 10])
    loss_function = multiclass_criterion

method = 'apnmlal'
name = method + "-" + dataset.name
# ("mnist" if isinstance(dataset, DatasetMNIST) else ("checkerboard2x2" if isinstance(dataset, DatasetCheckerboard2x2) else "checkerboard4x4"))
metrics = Metrics(name, 'apnmlal')

accuracies = []
for experiment in range(experiments):

    device = "cpu"
    print(f"Using device: {device}")
    # Reset the dataset
    dataset.set_start_state_torch(len(classes))
    dataset.trainData.grad = None
    dataset.trainLabels.grad = None

    # Start tracking fresh data
    metrics.new_experiment()
    # Randomly initialize the model weights
    fit_model.apply(init_model)
    model.apply(init_model)

    model.to(device)
    fit_model.to(device)

    dataset.testData = dataset.testData.to(device)
    dataset.testLabels = dataset.testLabels.to(device)
    dataset.trainLabels = dataset.trainLabels.to(device)
    dataset.trainData = dataset.trainData.to(device)

    for iteration in range(iterations):
        # 1. Train the model
        # 2. Evaluate the model
        # 3. Select the next point
        start = time.time()
        known_data = dataset.trainData[dataset.indicesKnown, :]
        known_labels = dataset.trainLabels[dataset.indicesKnown, :]
        known_data = known_data.to(device)
        known_labels = known_labels.to(device)
        print(known_data)
        print(known_labels)
        print("Running fit...")
        fit(fit_model, *fit_params, lambda m: loss_function(m(known_data), known_labels), early_stopping_patience=40)
        print("Running metrics.evaluate...")
        metrics.evaluate(fit_model, dataset, loss_function)
        print(f"Iteration {iteration}: {metrics.validation_accuracy[-1][-1]}")
        selectNext()

        end = time.time()
        print(f"Experiment {experiment + 1} Iteration {iteration + 1} complete")
        print(f"Time: {end - start}")
    metrics.save()
