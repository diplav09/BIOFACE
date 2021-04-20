from torch.utils.data import Dataset
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import torch
import os

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

class LoadData(Dataset):

  def __init__(self, dataset_dir = './zx_7_d10_inmc_celebA_01.hdf5', test=False):

    filename = dataset_dir
    with h5py.File(filename, "r") as f:
      # List all groups
      a_group_key = list(f.keys())[0]
      # Get the data
      self.data = list(f[a_group_key])        

    self.dataset_size = len(self.data)

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, idx):
    itm = self.data[idx]
    image = np.asarray(itm[0:3,:,:])

    diffuse = np.asarray(itm[3:4,:,:])

    mask =  np.asarray(itm[6:7,:,:])

    return image, diffuse, mask