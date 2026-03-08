import torch
from torch.utils.data import TensorDataset, DataLoader


def precompute_features(model, dataset, device):
    """
    Create a new dataset with the features precomputed by the model.

    If the model is f o g where f is the last layer and g is the rest
    of the model, we precompute g(x) for all x in the dataset and return
    a new dataset {(g(x_n), y_n)}.

    Arguments:
    ----------
    model: torchvision.models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    model = model.to(device)
    model.eval()

    features_list = []
    labels_list = []

    # On utilise un hook sur avgpool pour capturer les features juste avant fc
    activation = {}

    def hook_fn(module, input, output):
        activation["features"] = output

    handle = model.avgpool.register_forward_hook(hook_fn)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)
            feats = activation["features"].flatten(1)  # (batch, 512)
            features_list.append(feats.cpu())
            labels_list.append(labels)

    handle.remove()

    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return TensorDataset(all_features, all_labels)
