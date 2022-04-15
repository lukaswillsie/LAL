import copy

import torch
from numpy import ndarray
from sklearn import clone
from torch import nn, Tensor, optim
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class Model:
    """
    An interface for an ML model that mimics that of sklearn. The code was originally written to be used exclusively for
    RandomForestClassifiers, but we wanted to try out other model classes, so we introduced this quick and dirty
    interface to prevent having to change the existing code too much.
    """
    def fit(self, X: ndarray, y: ndarray):
        """
        Fit the model on the given data and return self. Both inputs should be 2-dimensional:

        X: num_examples x input_dim
        y: num_examples x 1
        """
        raise NotImplementedError

    def predict(self, X: ndarray) -> ndarray:
        """
        Use the current state of the model's parameters to predict class labels
        for all examples in X

        X: num_examples x input_dim

        Output: num_examples x 1
        """
        raise NotImplementedError

    def predict_proba(self, X: ndarray):
        """
        Use the current state of the model's parameters to make class probability
        predictions for each example in X

        X: num_examples x input_dim

        Output: num_examples x output_classes
        """
        raise NotImplementedError

    def clone(self):
        """
        Return a copy of the model with the same parameters, not fitted on data yet
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the model (re-initialize the weights) for a new experiment
        """
        pass


class SKLearnModel(Model):
    def __init__(self, model):
        self.model = model

    def fit(self, X: ndarray, y: ndarray):
        # sklearn wants the labels in a 1-d array
        return self.model.fit(X, y.flatten())

    def predict(self, X) -> ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: ndarray) -> ndarray:
        return self.model.predict_proba(X)

    def clone(self):
        return clone(self.model)


class PyTorchModel(Model):
    def __init__(
            self,
            model: nn.Module,
            epochs,
            lr,
            lr_decay=0.99,
            early_stopping_patience=None
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.early_stopping_patience = early_stopping_patience
        self.total_params = model.total_params
        self.standard_loss = BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr
        )

    def set_parameters(self, new_parameters: Tensor):
        self.model.set_parameters(new_parameters)

    def get_parameters(self):
        return self.model.get_parameters()

    @staticmethod
    def binary_criterion(m, inputs, labels):
        return BCEWithLogitsLoss()(m.forward(inputs).view(-1), labels.float())

    @staticmethod
    def multiclass_criterion(m, inputs, labels):
        return CrossEntropyLoss()(m.forward(inputs), labels.long())

    # <criterion> should be a callable that takes in the model outputs
    # and labels, each as flattened arrays, and returns a differentiable
    # loss
    def fit(self, X: ndarray, y: ndarray, criterion=None):
        criterion = criterion \
            if criterion else \
            (self.binary_criterion if self.model.out_size == 1 else self.multiclass_criterion)
        best_loss = float("inf")
        early_stop = 0
        train_data = torch.from_numpy(X).float()
        train_labels = torch.from_numpy(y)
        train_labels = train_labels.view(-1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            train_loss = criterion(self.model, train_data, train_labels)

            train_loss.backward()
            self.optimizer.step()

            if self.early_stopping_patience:
                if train_loss.item() < best_loss:
                    early_stop = 0
                    best_loss = train_loss.item()
                else:
                    early_stop += 1

                if early_stop > self.early_stopping_patience:
                    return self

        return self

    def predict(self, X) -> ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: ndarray) -> ndarray:
        with torch.no_grad():
            # Binary classification via sigmoid
            if self.model.out_size == 1:
                positive_prob = torch.sigmoid(self.model.forward(torch.from_numpy(X).float()))
                prob = torch.cat((1 - positive_prob, positive_prob), dim=1)
            else:
                prob = torch.softmax(self.model(torch.from_numpy(X).float()), dim=1)
            return prob.numpy()

    def clone(self):
        return PyTorchModel(
            copy.deepcopy(self.model),
            self.epochs,
            self.lr,
            self.lr_decay,
            self.early_stopping_patience
        )

    def reset(self):
        for p in self.model.parameters():
            torch.nn.init.normal_(p)


def _shape_size(size: torch.Size):
    mult = 1
    for dim in size:
        mult *= dim
    return mult


class ACNMLNet(nn.Module):
    def __init__(self):
        super(ACNMLNet, self).__init__()
        self.total_params = 0

    def count_parameters(self):
        for p in self.parameters():
            self.total_params += _shape_size(p.data.shape)

    def set_parameters(self, new_parameters: Tensor):
        assert len(new_parameters.shape) == 1
        assert new_parameters.shape[0] == self.total_params
        # Create a clone of the parameters instead of pointing directly to the input Tensor
        new_parameters = new_parameters.detach().clone()
        offset = 0
        for param in self.parameters():
            shape_size = _shape_size(param.data.shape)
            param.data = new_parameters[offset:offset + shape_size].reshape(param.data.shape)
            offset += shape_size

    def get_parameters(self):
        params = torch.zeros(self.total_params)
        offset = 0
        for param in self.parameters():
            shape_size = _shape_size(param.data.shape)
            params[offset:offset + shape_size] = param.data.view(-1).detach()
            offset += shape_size
        return params


class SimpleMLP(ACNMLNet):
    def __init__(self, sizes):
        super(SimpleMLP, self).__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.out_size = sizes[-1]
        self.count_parameters()

    def forward(self, x):
        return self.layers(x)
