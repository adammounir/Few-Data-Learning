import torch.nn as nn


class LastLayer(nn.Module):
    """
    Derniere couche du ResNet adaptee a notre probleme de classification binaire.
    On remplace la couche fc originale (512 -> 1000) par (512 -> 2).
    """

    def __init__(self, in_features=512, out_features=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
