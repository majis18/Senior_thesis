import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from sklearn.metrics import cohen_kappa_score


class Net_for_classification2(nn.Module):
    def __init__(self, n_1, n_2, num_classes):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.fc = nn.Linear(n_2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_regression2(nn.Module):
    def __init__(self, n_1, n_2):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.fc = nn.Linear(n_2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_classification3(nn.Module):
    def __init__(self, n_1, n_2, n_3, num_classes):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.fc = nn.Linear(n_3, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_regression3(nn.Module):
    def __init__(self, n_1, n_2, n_3):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.fc = nn.Linear(n_3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self,data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Net_for_classification4(nn.Module):
    def __init__(self, n_1, n_2, n_3, n_4, num_classes):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.conv4 = GCNConv(n_3, n_4)
        self.fc = nn.Linear(n_4, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_regression4(nn.Module):
    def __init__(self, n_1, n_2, n_3, n_4):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.conv4 = GCNConv(n_3, n_4)
        self.fc = nn.Linear(n_4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_classification5(nn.Module):
    def __init__(self, n_1, n_2, n_3, n_4, n_5, num_classes):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.conv4 = GCNConv(n_3, n_4)
        self.conv5 = GCNConv(n_4, n_5)
        self.fc = nn.Linear(n_5, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = self.conv5(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_regression5(nn.Module):
    def __init__(self, n_1, n_2, n_3, n_4, n_5):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.conv4 = GCNConv(n_3, n_4)
        self.conv5 = GCNConv(n_4, n_5)
        self.fc = nn.Linear(n_5, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = self.conv5(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_classification6(nn.Module):
    def __init__(self, n_1, n_2, n_3, n_4, n_5, n_6, num_classes):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.conv4 = GCNConv(n_3, n_4)
        self.conv5 = GCNConv(n_4, n_5)
        self.conv6 = GCNConv(n_5, n_6)
        self.fc = nn.Linear(n_6, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = self.conv5(x, edge_index)
        x = self.relu(x)
        x = self.conv6(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_regression6(nn.Module):
    def __init__(self, n_1, n_2, n_3, n_4, n_5, n_6):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.conv4 = GCNConv(n_3, n_4)
        self.conv5 = GCNConv(n_4, n_5)
        self.conv6 = GCNConv(n_5, n_6)
        self.fc = nn.Linear(n_6, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = self.conv5(x, edge_index)
        x = self.relu(x)
        x = self.conv6(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Net_for_regression3(nn.Module):
    def __init__(self, n_1, n_2, n_3):
        super().__init__()
        self.conv1 = GCNConv(-1, n_1)
        self.conv2 = GCNConv(n_1, n_2)
        self.conv3 = GCNConv(n_2, n_3)
        self.fc = nn.Linear(n_3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self,data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# class Net_GAT(nn.Module):
#     def __init__(self, n_h, n_head):
#         super().__init__()
#         self.gat1 = GATConv(-1, n_h, heads=n_head)
#         self.gat2 = GATConv(n_h*n_head, n_h, heads=n_head)
#         self.fc = nn.Linear(n_h*n_head, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.25)

#     def forward(self, data):
#         x = data.x
#         edge_index = data.edge_index
#         batch = data.batch
#         x = self.gat1(x, edge_index)
#         x = self.relu(x)
#         x = self.gat2(x, edge_index)
#         x = self.relu(x)
#         x = global_mean_pool(x, batch)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x


class KappaLoss(nn.Module):
    def __init__(self):
        super(KappaLoss, self).__init__()
    def forward(self, output, target):
        k = cohen_kappa_score(target, output, weights='quadratic')
        return k
    
def loss_fnc(out, y, criterion):
    if criterion=='cross_entropy_loss':
        crit = nn.CrossEntropyLoss()
    if criterion=='mse':
        crit = nn.MSELoss()
    if criterion=='kappa':
        crit = KappaLoss()
    loss = crit(out, y)
    return loss