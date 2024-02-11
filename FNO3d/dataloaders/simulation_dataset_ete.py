import os
import  os.path
import  numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datetime import datetime

class SimulationDataset(Dataset):
    def __init__(self, data_folder, z_min=70, z_max=134, total_sample_number = None, transform = None, threshold=1e-4):
        # strategy: loading data on the fly (especially when data is too big that doesn't fit into memory)
        self.root_dir = data_folder
        self.directories = os.listdir(self.root_dir) # make sure that data_folder only stores directories of samples 
        print(f"total number of folders: {len(self.directories)}")
        self.threshold = threshold
        # num_converged = self.check_converged()
        # print(f"converged folders: {num_converged}")
        # assert num_converged == len(self.directories)

        self.z_min = z_min
        self.z_max = z_max

        dataset_means = {
            '/media/ps2/chenkaim/3d_data/periodic_grating_wl800_th600_TiO2': np.array([0.00037015, 0.00037738, 0.00017998, 0.00017892, 0.00022806, 0.00022944]),
            '/media/ps2/chenkaim/3d_data/periodic_grating_wl800_th600': np.array([0.00037922, 0.00039991, 0.00021897, 0.00021664, 0.00025441, 0.00026113])
        }
        if data_folder not in dataset_means:
        # run this once when a new dataset is used, then store the values
            self.means = self.get_means()
            print("ds.means = ", self.means)

        # then hardcode the values here:
        else:
            self.means = dataset_means[data_folder]


        if total_sample_number:
            random.seed(1234)
            indices = np.array(random.sample(list(range(len(self.directories))), total_sample_number))
            self.directories = [self.directories[i] for i in indices]

    def __len__(self):
        return len(self.directories)

    def check_converged(self):
        num_converged = 0
        for d in tqdm(self.directories):
            with open(self.root_dir + '/' +d+'/' + "solver_info.txt", 'r') as f:
                f.readline()
                error = f.readline()[1:-1].split(",")
                if float(error[-1]) < self.threshold:
                    num_converged += 1
        return num_converged

    def get_means(self):
        means = np.array([0.,0.,0.,0.,0.,0.], dtype = np.float32)
        for d in tqdm(self.directories):
            Ex = np.load(self.root_dir + '/' +d+'/'+'0.0001Ex.npy', mmap_mode='r').astype(np.complex64, copy=False)
            Ey = np.load(self.root_dir + '/' +d+'/'+'0.0001Ey.npy', mmap_mode='r').astype(np.complex64, copy=False)
            Ez = np.load(self.root_dir + '/' +d+'/'+'0.0001Ez.npy', mmap_mode='r').astype(np.complex64, copy=False)

            means += np.array([np.mean(np.abs(Ex.real)), np.mean(np.abs(Ex.imag)),\
                               np.mean(np.abs(Ey.real)), np.mean(np.abs(Ey.imag)),
                               np.mean(np.abs(Ez.real)), np.mean(np.abs(Ez.imag))])

        return means/len(self.directories)

    def __getitem__(self, idx):
        folder = self.directories[idx]
        yee = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'epsilon_DDM.npy', mmap_mode='r'), dtype=torch.float32)

        yeex, yeey, yeez = yee[0,:,:,self.z_min:self.z_max], yee[1,:,:,self.z_min:self.z_max], yee[2,:,:,self.z_min:self.z_max]
        # print(f"yee shapes: yeex: {yeex.shape}, yeey: {yeey.shape}, yeez: {yeez.shape}")

        Ex = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'0.0001Ex.npy', mmap_mode='r')[:,:,self.z_min:self.z_max], dtype=torch.complex64)
        Ey = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'0.0001Ey.npy', mmap_mode='r')[:,:,self.z_min:self.z_max], dtype=torch.complex64)
        Ez = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'0.0001Ez.npy', mmap_mode='r')[:,:,self.z_min:self.z_max], dtype=torch.complex64)
        # print("Ex: ", Ex.shape, Ex.dtype)

        # apply a random roll:
        rx, ry = np.random.randint(yeex.shape[0]), np.random.randint(yeex.shape[1])
        yeex = torch.roll(torch.roll(yeex, rx, dims=0), ry, dims=1)
        yeey = torch.roll(torch.roll(yeey, rx, dims=0), ry, dims=1)
        yeez = torch.roll(torch.roll(yeez, rx, dims=0), ry, dims=1)

        Ex = torch.roll(torch.roll(Ex, rx, dims=0), ry, dims=1)
        Ey = torch.roll(torch.roll(Ey, rx, dims=0), ry, dims=1)
        Ez = torch.roll(torch.roll(Ez, rx, dims=0), ry, dims=1)

        field = torch.stack((Ex.real/self.means[0], Ex.imag/self.means[1], Ey.real/self.means[2], Ey.imag/self.means[3], Ez.real/self.means[4], Ez.imag/self.means[5]), dim=-1)
        sample = {'field': field, 'yeex': yeex, 'yeey': yeey, 'yeez': yeez}

        return sample

