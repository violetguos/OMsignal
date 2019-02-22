import torch
from torch.utils.data import DataLoader
from src.utils import constants
from src.legacy.TABaseline.code.ecgdataset import ECGDataset
from torch import nn, optim
from torch.nn import functional as F
import sys
import configparser

"""Before doing a variational autoencoder, start with a basic autoencoder"""
# TODO:
# 1. use config parser
# 2. create DataSet class for unlabeled data, or find under legacy (JP is working on it)
# 3. read up on VAE
# 4. Log results for tensorboardX
# 5. Add loss on eval dataset if grid search is needed
# 6. Save model

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # the number of hidden units are hardcoded for now.
        self.encoder = nn.Sequential(
            nn.Linear(3750, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3750),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(epoch, model, optimizer, train_loader):
    model.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        print("data iter", data.shape)
        data = data.to(device)
        print("data iter to dev", data.shape)
        data = data.view(32, 3750)

        optimizer.zero_grad()
        output = model(data)

        BCE_loss = nn.BCELoss()(output, data)
        MSE_loss = nn.MSELoss()(output, data)

        # ===================backward====================
        optimizer.zero_grad()
        MSE_loss.backward()
        optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, batch_idx, BCE_loss.data, MSE_loss.data))

    return MSE_loss.item()



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('src/algorithm/base_AE_input.in') #sys.argv[1])
    train_dataset = ECGDataset(
        constants.TRAIN_LABELED_DATASET_PATH, True, target=constants.TARGETS)
    valid_dataset = ECGDataset(
        constants.VALID_LABELED_DATASET_PATH, False, target=constants.TARGETS)
    batch_size = int(config.get('optimizer', 'batch_size'))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False,
                              num_workers=1)

    device = torch.device('cuda')
    model = autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    nepoch = int(config.get('optimizer', 'nepoch'))

    train_loss_history = []
    for epoch in range(nepoch):
        train_loss_history.append(train(epoch, model, optimizer, train_loader))
