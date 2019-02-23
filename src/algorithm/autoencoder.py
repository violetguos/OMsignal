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

# TODO:
# -Done 1. use config parser
# -Done 2. create DataSet class for unlabelled data, or find under legacy


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.preprocess = pp.Preprocessor()

        # the number of hidden units are hardcoded for now.
        self.encoder = nn.Sequential(
            nn.Linear(3750, 1024), nn.ReLU(True), nn.Linear(1024, 256), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024), nn.ReLU(True), nn.Linear(1024, 3750)
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(epoch, model, optimizer, batch_size, train_loader):
    model.train()
    total_batch = constants.UNLABELED_SHAPE[0] // batch_size

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(batch_size, 1, 3750)
        # print("data iter to dev", data)

        optimizer.zero_grad()
        output = model(data)


        data = pp.Preprocessor().forward(data)
        # print("data after prepro\n")
        # print(data)
        MSE_loss = nn.MSELoss()(output, data)

        # ===================backward====================
        optimizer.zero_grad()
        MSE_loss.backward()
        optimizer.step()
        print(
            "epoch [{}], batch [{}/{}], MSE_loss:{:.4f}".format(
                epoch+1, batch_idx, total_batch, MSE_loss.data
            )
        )
        # TODO: make a helper function under util/cache.py and use a name generator for the model
        # torch.save(model.state_dict(), "../../model")
    return MSE_loss.item()

def plot_signal(output):
    print(output)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("src/algorithm/autoencoder_input.in")
    train_dataset = UnlabelledDataset(constants.UNLABELED_DATASET_PATH, False)
    print(train_dataset)
    # read from config files
    batch_size = int(config.get("optimizer", "batch_size"))
    print(type(batch_size))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)

    device = torch.device("cpu")
    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    nepoch = int(config.get('optimizer', 'nepoch'))
    train_loss_history = []
    for epoch in range(nepoch):
        train_loss_history.append(train_autoencoder(epoch, model, optimizer, 32, train_loader))
