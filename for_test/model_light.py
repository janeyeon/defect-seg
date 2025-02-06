import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConNet(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024):
        super(SupConNet, self).__init__()
        # self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()
        self.encoder = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.BatchNorm1d(input_dim),
        nn.ReLU()
        # nn.Linear(hidden_dim, hidden_dim // 2),
        # nn.BatchNorm1d(hidden_dim // 2),
        # nn.ReLU(),
        # nn.Linear(hidden_dim // 2, input_dim)  # Return to original dimension
    )
        
    def forward(self, x):
        # with torch.no_grad():
        #     features = self.dinov2(x)
        feat = self.encoder(x)
        feat = F.normalize(feat, p=2, dim=1)
        # feat = self.model(x)
       # feat = F.normalize(feat, p=2, dim=1)  # L2 Normalization
        return feat



# # Set the root directory of your dataset (replace with your actual directory)
# root_dir = '/home/jeeit18/b1/DefectSeg/data/Vision_v1_pre/train'  # e.g., './data'

# # Create dataset and dataloader
# dataset = CustomDataset(root_dir=root_dir)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Example of iterating through the dataloader
# for images, labels in dataloader:
#     print(images.shape, labels)
