from torch import nn, optim
from src.legacy.TABaseline.code import Preprocessor as pp


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
