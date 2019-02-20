import pickle
import torch

if __name__ == '__main__':
    model = torch.load('../../model/baseline_final.pt', map_location=torch.device('cpu'))

    print(model)