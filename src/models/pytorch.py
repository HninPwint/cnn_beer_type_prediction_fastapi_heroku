import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device


class PytorchClassification_1(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layer_1 = nn.Linear(n_features, 256)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.layer_2 = nn.Linear(256, 128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.layer_out = nn.Linear(128, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class PytorchClassification_2(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layer_1 = nn.Linear(n_features, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.layer_2 = nn.Linear(512, 128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.layer_3 = nn.Linear(128, 64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.layer_out = nn.Linear(64, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class PytorchClassification_5(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layer_1 = nn.Linear(n_features, 1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.layer_2 = nn.Linear(1024, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.layer_3 = nn.Linear(256, 128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.layer_4 = nn.Linear(128, 64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.layer_out = nn.Linear(64, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class PytorchClassification_6(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layer_1 = nn.Linear(n_features, 2048)
        self.batchnorm1 = nn.BatchNorm1d(2048)
        self.layer_2 = nn.Linear(2048, 512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.layer_3 = nn.Linear(512, 256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.layer_4 = nn.Linear(256, 64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.layer_out = nn.Linear(64, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class PytorchClassification_8(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layer_1 = nn.Linear(n_features, 4096)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.layer_2 = nn.Linear(4096, 1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.layer_3 = nn.Linear(1024, 256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.layer_4 = nn.Linear(256, 128)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.layer_5 = nn.Linear(128, 64)
        self.batchnorm5 = nn.BatchNorm1d(64)
        self.layer_out = nn.Linear(64, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x
    

class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...
    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensofr
    y_tensor : Pytorch tensor
        Target tensor
    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """

    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)

    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]

    def __len__(self):
        return len(self.X_tensor)

    def to_tensor(self, data):
        return torch.Tensor(np.array(data))


def train_classification(train_data,
                         model,
                         criterion,
                         optimizer,
                         batch_size,
                         device,
                         scheduler=None,
                         generate_batch=None):
    """Train a Pytorch binary classification model
    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps
    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """

    # Set model to training mode
    model.train()
    train_loss = 0
    train_acc = 0

    # Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_batch)

    # Iterate through data by batch of observations
    for i, (feature, target_class) in enumerate(data):
        # Reset gradients
        optimizer.zero_grad()

        # Load data to specified device target_class
        feature, target_class = feature.to(device), target_class.to(device).to(torch.long)

        # Make predictions
        output = model(feature)

        # Calculate loss for given batch
        loss = criterion(output, target_class)

        # Calculate global loss
        train_loss += loss.item()

        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        # Calculate global accuracy
        train_acc += (output.argmax(1) == target_class).sum().item()

    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


def test_classification(test_data, model, criterion, batch_size, device,
                generate_batch=None):
    """Calculate performance of a Pytorch binary classification model
    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    batch_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps
    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """

    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    test_acc = 0

    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size,
                      collate_fn=generate_batch)

    # Iterate through data by batch of observations
    for feature, target_class in data:
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device).to(
            torch.long)

        # Set no update to gradients
        with torch.no_grad():
            # Make predictions
            output = model(feature)

            # Calculate loss for given batch
            loss = criterion(output, target_class)

            # Calculate global loss
            test_loss += loss.item()

            # Calculate global accuracy
            test_acc += (output.argmax(1) == target_class).sum().item()

    return test_loss / len(test_data), test_acc / len(test_data)