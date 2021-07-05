# %%
import os
os.chdir("/home/chw357/cse252d/cse252d-sp21-hw2/sphereFace")

import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
from torch.utils.data import DataLoader
import faceNet
import os
import numpy as np

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device is {device}")

# %%
# Initialize dataLoader
faceDataset = dataLoader.BatchLoader(
        imageRoot = '/datasets/cs252-sp21-A00-public/hw2_data/CASIA-WebFace',
        alignmentRoot = './data/casia_landmark.txt',
        cropSize = (96, 112)
        )
faceLoader = DataLoader(faceDataset, batch_size = 1, num_workers = 16, shuffle = False )

# %% load network
net_type = 'faceNet'
model_path = '/datasets/home/80/080/chw357/cse252d/cse252d-sp21-hw2/sphereFace/checkpoint/netFinal_8.pth'
net = getattr(faceNet, net_type)()
net.load_state_dict(torch.load(model_path))
net.to(device)
net.eval()
net.feature = True

# %%
sample_idxes = np.random.choice(np.arange(len(faceLoader)), size=10, replace=False)
net_outputs = {}

for face_idx in sample_idxes:
    img = faceLoader.dataset[face_idx]["img"]
    target = faceLoader.dataset[face_idx]["target"]
    
    img = img.to(device)
    net_outputs[target] = net(img)
