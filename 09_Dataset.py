import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math

class WineDataset(Dataset): 
    def __init__(self):
        # Data loading
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # n_samples, 1
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self): 
        return self.n_samples

if __name__ == "__main__":
    dataset = WineDataset()
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)  # num_workers=0 für Windows

    data_iter = iter(data_loader)
    data = next(data_iter)
    features, labels = data 

    print(features, labels)
