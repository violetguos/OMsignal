import torch
from torch.utils.data import DataLoader
from src.utils import constants
from src.legacy.TABaseline.code.ecgdataset import ECGDataset
from torch import nn, optim
from torch.nn import functional as F
from src.data.unlabelled_data import UnlabelledDataset
import configparser
import sys
from src.legacy.TABaseline.code import Preprocessor as pp


"""Before doing a variational autoencoder, start with a basic autoencoder"""
# TODO:
# -Done 1. use config parser
# -Done 2. create DataSet class for unlabelled data, or find under legacy


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.preprocess = pp.Preprocessor()

        # the number of hidden units are hardcoded for now.
        self.encoder = nn.Sequential(
            nn.Linear(3750, 1024), nn.ReLU(True), nn.Linear(1024, 256), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024), nn.ReLU(True), nn.Linear(1024, 3750), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(epoch, model, optimizer, train_loader):
    model.train()
    total_batch = constants.UNLABELED_SHAPE[0] // 32

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(32, 1, 3750)
        # print("data iter to dev", data)

        optimizer.zero_grad()
        output = model(data)

        data = pp.Preprocessor().forward(data)
        print("data after prepro\n")
        print(data)
        BCE_loss = nn.BCELoss()(output, data)
        MSE_loss = nn.MSELoss()(output, data)

        # ===================backward====================
        optimizer.zero_grad()
        MSE_loss.backward()
        optimizer.step()
        print(
            "batch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}".format(
                batch_idx, total_batch, BCE_loss.data, MSE_loss.data
            )
        )
        # TODO: make a helper function under util/cache.py and use a name generator for the model
        # torch.save(model.state_dict(), "../../model")
    return MSE_loss.item()


def plot_signal(output):
    print(output)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("base_AE_input.in")
    train_dataset = UnlabelledDataset(constants.T5_FAKE_VALID_LABELED_DATA, False)
    print(train_dataset)
    # read from config files
    batch_size = int(config.get("optimizer", "batch_size"))
    print(type(batch_size))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)

    device = torch.device("cuda")
    model = autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    nepoch = int(config.get('optimizer', 'nepoch'))
    train_loss_history = []
    for epoch in range(nepoch):
        train_loss_history.append(train(epoch, model, optimizer, train_loader))
