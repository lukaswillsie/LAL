import pickle
import numpy as np
import torch
from Classes.dataset import Dataset
from torch import optim, nn
import matplotlib.pyplot as plt


class Metrics:
    metrics = {
        "validation_accuracy",
        "validation_loss",
        "average_uncertainty",
        "average_uncertainty_correct",
        "average_uncertainty_incorrect"
    }

    def __init__(self, name, method):
        self.name = name
        self.method = method
        self.experiment_num = -1
        #  For each experiment we will keep a list of lists
        self.validation_accuracy = []
        self.validation_loss = []
        self.average_uncertainty = []
        self.average_uncertainty_correct = []
        self.average_uncertainty_incorrect = []

    def evaluate(self, model, dataset: Dataset, loss_function):
        """
        loss_function should be a callable that takes the model's raw output and the
        labels

        Metrics being tracked:
        1. Validation accuracy
        2. Validation loss
        3. Average Uncertainty
        4. Average Uncertainty on Correct Predictions
        5. Average Uncertainty on Incorrect Predictions
        """
        with torch.no_grad():
            #  Raw output of the model
            output = model(dataset.testData)
            if dataset.is_binary:
                # Sigmoid classification boundary is at 0
                as_class = (output >= 0).long()
                pos_prob = torch.sigmoid(output)
                probabilities = torch.cat((1 - pos_prob, pos_prob), dim=1)
            else:
                probabilities = torch.softmax(output, dim=1)
                as_class = torch.argmax(output, dim=1, keepdim=True)

            correct = as_class == dataset.testLabels.long()
            self.validation_accuracy[self.experiment_num].append((torch.sum(correct) / output.shape[0]).item())
            self.validation_loss[self.experiment_num].append(loss_function(output, dataset.testLabels))

            # For each example, compute the entropy of the predicted distribution
            uncertainty = -torch.nansum(
                probabilities * torch.log(probabilities),
                dim=1,
                keepdim=True
            )
            self.average_uncertainty[self.experiment_num].append(torch.mean(uncertainty))
            self.average_uncertainty_correct[self.experiment_num].append(torch.mean(uncertainty[correct]))
            self.average_uncertainty_incorrect[self.experiment_num].append(torch.mean(uncertainty[~correct]))

    def new_experiment(self):
        self.experiment_num += 1
        self.validation_accuracy.append([])
        self.validation_loss.append([])
        self.average_uncertainty.append([])
        self.average_uncertainty_correct.append([])
        self.average_uncertainty_incorrect.append([])

    def save(self):
        pickle.dump(self, open(f"metrics/{self.name}.pkl", "wb"))

    def plot(self):
        for metric in self.metrics:
            plt.plot(np.mean(np.array(self.__dict__[metric]), axis=0))
            plt.xlabel('# labelled points')
            plt.title(metric)
            plt.savefig(f"./new-out/{self.name}-{metric}.png")
            plt.clf()


def init_model(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight)
        torch.nn.init.normal_(m.bias)


def plot_combined_metrics(*metrics):
    for metric in Metrics.metrics:
        plt.title(metric)
        plt.xlabel('# labelled points')
        methods = []
        for metric_obj in metrics:
            methods.append(metric_obj.method)
            plt.plot(np.mean(np.array(metric_obj.__dict__[metric]), axis=0), label=metric_obj.method)
        plt.legend(loc='lower right')
        plt.savefig(f"./new-out/combined-{metric}-{'-'.join(methods)}.png")
        plt.clf()


def predict_probabilities(m, inputs):
    with torch.no_grad():
        pred = m(inputs)
        # Binary classification with logits
        if pred.shape[-1] == 1:
            pos_prob = torch.sigmoid(pred)
            return torch.cat((1 - pos_prob, pos_prob), dim=1)
        # Softmax
        else:
            return torch.softmax(pred, dim=1)


def fit(target_model, epochs, lr, criterion, early_stopping_patience=None, debug=False):
    """
    Criterion should be a callable that accepts JUST THE MODEL. For example:
    def criterion(model):
        return CrossEntropyLoss()(model(inputs), labels)

    fit(model, 42, 1e-42, criterion)

    This is because the ACNML target is a function of the parameters of the model, not just of the outputs and the labels
    """
    optimizer = optim.Adam(
        target_model.parameters(), lr=lr
    )
    best_loss = float("inf")
    early_stop = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        train_loss = criterion(target_model)
        if debug:
            print(train_loss)
        train_loss.backward()
        optimizer.step()

        if early_stopping_patience:
            if train_loss.item() < best_loss:
                early_stop = 0
                best_loss = train_loss.item()
            else:
                early_stop += 1

            if early_stop > early_stopping_patience:
                if debug:
                    print(f"Early stopping after {epoch+1} iterations")
                return

    return
