import torch
from Classes.dataset import Dataset
from torch import optim


class Metrics:
    def __init__(self):
        self.data = {}


def evaluate(model, dataset: Dataset):
    with torch.no_grad():
        pred = model(dataset.testData)
        if dataset.is_binary:
            pred = torch.sigmoid(pred)
            pred = (pred >= 0.5).long()
        else:
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1, keepdim=True)
        return (torch.sum(pred == dataset.testLabels.long()) / pred.shape[0]).item()


def predict_probabilities(m, inputs):
    with torch.no_grad():
        pred = m(inputs)
        # Binary classification with logits
        if pred.shape[0] == 1:
            pos_prob = torch.sigmoid(pred)
            return torch.cat((1 - pos_prob, pos_prob), dim=1)
        # Softmax
        else:
            return torch.softmax(pred, dim=1)


def fit(target_model, epochs, lr, criterion, early_stopping_patience=None):
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
