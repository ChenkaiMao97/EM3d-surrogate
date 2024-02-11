import torch
from torch.utils.data import IterableDataset
import webdataset as wds
import  numpy as np

class SimulationDataset(IterableDataset):
    def __init__(self, data_path, means=None, z_min=45, z_max=138, cube_size=64, transform = None, wl=800e-9, dL=12.5e-9):
        super().__init__()
        # Initialize the WebDataset with the pattern matching your tar files
        self.dataset = wds.WebDataset(data_path, resampled=True).decode("torch")
        self.z_min = z_min
        self.z_max = z_max
        self.cube_size = cube_size
        self.wl = wl
        self.dL = dL

        self.means = means

    def __iter__(self):
        for sample in self.dataset:
            # Extracting the .npy files for each sample
            eps, Ex, Ey, Ez = sample['epsilon_ddm.npy'], sample['0.0001ex.npy'], sample['0.0001ey.npy'], sample['0.0001ez.npy']
            
            yee = torch.from_numpy(eps).type(torch.float32)
            Ex = torch.from_numpy(Ex).type(torch.complex64)
            Ey = torch.from_numpy(Ey).type(torch.complex64)
            Ez = torch.from_numpy(Ez).type(torch.complex64)

            yeex, yeey, yeez = yee[0], yee[1], yee[2]
            # x, y starting point could be random
            rx, ry = np.random.randint(yeex.shape[0]), np.random.randint(yeex.shape[1])

            # z starting point is in range of (z_min, z_max-cube_size)
            rz = np.random.randint(self.z_min, self.z_max-self.cube_size)
            # apply a random roll:
            
            yeex = torch.roll(torch.roll(torch.roll(yeex, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:self.cube_size,:self.cube_size,:self.cube_size]
            yeey = torch.roll(torch.roll(torch.roll(yeey, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:self.cube_size,:self.cube_size,:self.cube_size]
            yeez = torch.roll(torch.roll(torch.roll(yeez, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:self.cube_size,:self.cube_size,:self.cube_size]

            Ex = torch.roll(torch.roll(torch.roll(Ex, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:self.cube_size,:self.cube_size,:self.cube_size]
            Ey = torch.roll(torch.roll(torch.roll(Ey, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:self.cube_size,:self.cube_size,:self.cube_size]
            Ez = torch.roll(torch.roll(torch.roll(Ez, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:self.cube_size,:self.cube_size,:self.cube_size]

            field = torch.stack((Ex.real/self.means[0], Ex.imag/self.means[1], Ey.real/self.means[2], Ey.imag/self.means[3], Ez.real/self.means[4], Ez.imag/self.means[5]), dim=-1)

            # top_bc0 =    1j*2*np.pi*np.sqrt(yeex[1:2, :])*self.dL/self.wl*1/2*(field_rot0[0:1, :]+field_rot0[1:2, :]) + field_rot0[0:1, :]-field_rot0[1:2, :]
            bc_x_p = torch.cat([torch.view_as_real((f[-1,: ,: ]-f[-2,: ,: ])+1/2*(f[-1,: ,: ]+f[-2,: ,: ])*1j*2*np.pi*np.sqrt(yeex[-1,: ,: ])*self.dL/self.wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(self.means[None,None,:])
            bc_x_n = torch.cat([torch.view_as_real((f[0 ,: ,: ]-f[1 ,: ,: ])+1/2*(f[0 ,: ,: ]+f[1 ,: ,: ])*1j*2*np.pi*np.sqrt(yeex[1 ,: ,: ])*self.dL/self.wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(self.means[None,None,:])
            bc_y_p = torch.cat([torch.view_as_real((f[: ,-1,: ]-f[: ,-2,: ])+1/2*(f[: ,-1,: ]+f[: ,-2,: ])*1j*2*np.pi*np.sqrt(yeey[: ,-1,: ])*self.dL/self.wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(self.means[None,None,:])
            bc_y_n = torch.cat([torch.view_as_real((f[: ,0 ,: ]-f[: ,1 ,: ])+1/2*(f[: ,0 ,: ]+f[: ,1 ,: ])*1j*2*np.pi*np.sqrt(yeey[: ,1 ,: ])*self.dL/self.wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(self.means[None,None,:])
            bc_z_p = torch.cat([torch.view_as_real((f[: ,: ,-1]-f[: ,: ,-2])+1/2*(f[: ,: ,-1]+f[: ,: ,-2])*1j*2*np.pi*np.sqrt(yeez[: ,: ,-1])*self.dL/self.wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(self.means[None,None,:])
            bc_z_n = torch.cat([torch.view_as_real((f[: ,: ,0 ]-f[: ,: ,1 ])+1/2*(f[: ,: ,0 ]+f[: ,: ,1 ])*1j*2*np.pi*np.sqrt(yeez[: ,: ,1 ])*self.dL/self.wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(self.means[None,None,:])

            bcs = torch.stack((bc_x_p, bc_x_n, bc_y_p, bc_y_n, bc_z_p, bc_z_n), dim = -2)

            sample = {'field': field, 'bcs': bcs, 'yeex': yeex, 'yeey': yeey, 'yeez': yeez}

            yield sample
