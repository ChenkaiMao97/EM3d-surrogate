import os
import  os.path
import  numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datetime import datetime

class SimulationDataset(Dataset):
    def __init__(self, data_folder, z_min=22, z_max=54, xpml=10, total_sample_number = None, transform = None, threshold=1e-4):
        # strategy: loading data on the fly (especially when data is too big that doesn't fit into memory)
        self.root_dir = data_folder
        self.directories = os.listdir(self.root_dir) # make sure that data_folder only stores directories of samples 
        print(f"total number of folders: {len(self.directories)}")
        self.threshold = threshold
        num_converged = self.check_converged()
        print(f"converged folders: {num_converged}")
        assert num_converged >= len(self.directories)-1 # if still generating, then should be # of folders -1

        self.xpml = xpml
        self.z_min = z_min
        self.z_max = z_max

        dataset_means = {
            '/media/lts0/chenkaim/3d_data/SR_half_periodic_version3': np.array([5.2912545e-04, 4.8816926e-04, 8.0871228e-05, 8.5572719e-05, 2.4360677e-04, 2.3910306e-04]),
            '/media/ps2/chenkaim/3d_data/SR_half_periodic_TiO2': np.array([5.3854572e-04, 5.0815661e-04, 9.2351920e-06, 1.0210312e-05, 1.2823625e-04, 1.2765337e-04]),
            '/media/lts0/chenkaim/3d_data/SR_half_periodic_TiO2': np.array([5.4153166e-04, 5.0744746e-04, 1.4698635e-05, 1.6196915e-05, 1.2937987e-04, 1.2893751e-04])
        }
        if data_folder not in dataset_means:
            self.means = self.get_means()
            print("ds.means = ", self.means)
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
            try:
                with open(self.root_dir + '/' +d+'/' + "solver_info.txt", 'r') as f:
                    f.readline()
                    error = f.readline()[1:-1].split(",")
                    if float(error[-1]) < self.threshold:
                        num_converged += 1
            except:
                continue
        return num_converged

    def get_means(self):
        means = np.array([0.,0.,0.,0.,0.,0.], dtype = np.float32)
        for d in tqdm(self.directories):
            try:
                Ex = np.load(self.root_dir + '/' +d+'/'+'0.0001Ex.npy', mmap_mode='r').astype(np.complex64, copy=False)[self.xpml:-self.xpml,:,self.z_min:self.z_max]
                Ey = np.load(self.root_dir + '/' +d+'/'+'0.0001Ey.npy', mmap_mode='r').astype(np.complex64, copy=False)[self.xpml:-self.xpml,:,self.z_min:self.z_max]
                Ez = np.load(self.root_dir + '/' +d+'/'+'0.0001Ez.npy', mmap_mode='r').astype(np.complex64, copy=False)[self.xpml:-self.xpml,:,self.z_min:self.z_max]

                means += np.array([np.mean(np.abs(Ex.real)), np.mean(np.abs(Ex.imag)),\
                                   np.mean(np.abs(Ey.real)), np.mean(np.abs(Ey.imag)),
                                   np.mean(np.abs(Ez.real)), np.mean(np.abs(Ez.imag))])
            except:
                continue

        return means/len(self.directories)

    def __getitem__(self, idx):
        folder = self.directories[idx]
        yee = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'epsilon_DDM.npy', mmap_mode='r'), dtype=torch.float32)

        yeex, yeey, yeez = yee[0,self.xpml:-self.xpml,:,self.z_min:self.z_max], yee[1,self.xpml:-self.xpml,:,self.z_min:self.z_max], yee[2,self.xpml:-self.xpml,:,self.z_min:self.z_max]
        # print(f"yee shapes: yeex: {yeex.shape}, yeey: {yeey.shape}, yeez: {yeez.shape}")

        Ex = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'0.0001Ex.npy', mmap_mode='r')[self.xpml:-self.xpml,:,self.z_min:self.z_max], dtype=torch.complex64)
        Ey = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'0.0001Ey.npy', mmap_mode='r')[self.xpml:-self.xpml,:,self.z_min:self.z_max], dtype=torch.complex64)
        Ez = torch.tensor(np.load(self.root_dir + '/' +folder+'/'+'0.0001Ez.npy', mmap_mode='r')[self.xpml:-self.xpml,:,self.z_min:self.z_max], dtype=torch.complex64)
        # print("Ex: ", Ex.shape, Ex.dtype)

        # apply a random roll:
        ry = np.random.randint(yeex.shape[1])
        yeex = torch.roll(yeex, ry, dims=1)
        yeey = torch.roll(yeey, ry, dims=1)
        yeez = torch.roll(yeez, ry, dims=1)

        Ex = torch.roll(Ex, ry, dims=1)
        Ey = torch.roll(Ey, ry, dims=1)
        Ez = torch.roll(Ez, ry, dims=1)

        field = torch.stack((Ex.real/self.means[0], Ex.imag/self.means[1], Ey.real/self.means[2], Ey.imag/self.means[3], Ez.real/self.means[4], Ez.imag/self.means[5]), dim=-1)
        # field = torch.stack((Ex.real/self.means[0], Ex.imag/self.means[1], Ez.real/self.means[4], Ez.imag/self.means[5]), dim=-1)
        sample = {'field': field, 'yeex': yeex, 'yeey': yeey, 'yeez': yeez}

        return sample

