import torch
from torch import nn, Tensor


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
