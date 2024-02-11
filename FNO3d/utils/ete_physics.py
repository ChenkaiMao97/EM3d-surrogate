import  torch
import  numpy as np

C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
dL = 12.5e-9
wavelength = 800e-9
omega = 2 * np.pi * C_0 / wavelength
# n_air=1.
# n_Si=3.82
# n_sub=1.45

########### physics, numpy, CPU ###########

def E_to_H(Ex, Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=None):
    Hx = E_to_Hx(Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=bloch_vector)
    Hy = E_to_Hy(Ez, Ex, dL, omega, MU_0 = MU_0, bloch_vector=bloch_vector)
    Hz = E_to_Hz(Ex, Ey, dL, omega, MU_0 = MU_0, bloch_vector=bloch_vector)
    return (Hx, Hy, Hz)

def E_to_Hx(Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=None):
    # dEzdy = Ez[:,1:,0:-1]-Ez[:,0:-1,0:-1]
    # dEydz = Ey[:,0:-1,1:]-Ey[:,0:-1,0:-1]
    # Hx = (dEzdy - dEydz) / (-1j*dL*omega*MU_0)
    # return Hx

    if bloch_vector is None:
        dEzdy = np.roll(Ez, -1, axis=1) - Ez # np.roll([1,2,3],-1) = [2,3,1]
        dEydz = np.roll(Ey, -1, axis=2) - Ey
    else:
        dEzdy = np.concatenate((Ez[:,1:,:], Ez[:,0:1,:]*np.exp(-1j*bloch_vector[1]*dL*1e9*Ez.shape[1])), axis=1) - Ez
        dEydz = np.concatenate((Ey[:,:,1:], Ey[:,:,0:1]*np.exp(-1j*bloch_vector[2]*dL*1e9*Ey.shape[2])), axis=2) - Ey

    Hx = (dEzdy - dEydz) / (-1j*dL*omega*MU_0)
    return Hx

def E_to_Hy(Ez, Ex, dL, omega, MU_0 = MU_0, bloch_vector=None):
    # dExdz = Ex[0:-1,:,1:]-Ex[0:-1,:,0:-1]
    # dEzdx = Ez[1:,:,0:-1]-Ez[0:-1,:,0:-1]
    # Hy = (dExdz - dEzdx) / (-1j*dL*omega*MU_0)
    # return Hy
    if bloch_vector is None:
        dExdz = np.roll(Ex, -1, axis=2) - Ex
        dEzdx = np.roll(Ez, -1, axis=0) - Ez
    else:
        dExdz = np.concatenate((Ex[:,:,1:], Ex[:,:,0:1]*np.exp(-1j*bloch_vector[2]*dL*1e9*Ex.shape[2])), axis=2) - Ex
        dEzdx = np.concatenate((Ez[1:,:,:], Ez[0:1,:,:]*np.exp(-1j*bloch_vector[0]*dL*1e9*Ez.shape[0])), axis=0) - Ez

    Hy = (dExdz - dEzdx) / (-1j*dL*omega*MU_0)
    return Hy

def E_to_Hz(Ex, Ey, dL, omega, MU_0 = MU_0, bloch_vector=None):
    # dEydx = Ey[1:,0:-1,:]-Ey[0:-1,0:-1,:]
    # dExdy = Ex[0:-1,1:,:]-Ex[0:-1,0:-1,:]
    # Hz = (dEydx - dExdy) / (-1j*dL*omega*MU_0)
    # return Hz
    if bloch_vector is None:
        dEydx = np.roll(Ey, -1, axis=0) - Ey
        dExdy = np.roll(Ex, -1, axis=1) - Ex
    else:
        dEydx = np.concatenate((Ey[1:,:,:], Ey[0:1,:,:]*np.exp(-1j*bloch_vector[0]*dL*1e9*Ey.shape[0])), axis=0) - Ey
        dExdy = np.concatenate((Ex[:,1:,:], Ex[:,0:1,:]*np.exp(-1j*bloch_vector[1]*dL*1e9*Ex.shape[1])), axis=1) - Ex

    Hz = (dEydx - dExdy) / (-1j*dL*omega*MU_0)
    return Hz

def H_to_E(Hx, Hy, Hz, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    Ex = H_to_Ex(Hy, Hz, dL, omega, eps_grid[0], EPSILON_0 = EPSILON_0, bloch_vector=bloch_vector)
    Ey = H_to_Ey(Hz, Hx, dL, omega, eps_grid[1], EPSILON_0 = EPSILON_0, bloch_vector=bloch_vector)
    Ez = H_to_Ez(Hx, Hy, dL, omega, eps_grid[2], EPSILON_0 = EPSILON_0, bloch_vector=bloch_vector)
    return (Ex, Ey, Ez)

def H_to_Ex(Hy, Hz, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    # dHzdy = Hz[:,1:,1:]-Hz[:,0:-1,1:]
    # dHydz = Hy[:,1:,1:]-Hy[:,1:,0:-1]
    # Ex = (dHzdy - dHydz) / (1j*dL*omega*EPSILON_0*eps_grid[:,1:,1:])
    # return Ex
    if bloch_vector is None:
        dHzdy = -np.roll(Hz, 1, axis=1) + Hz # np.roll([1,2,3],1) = [3,1,2]
        dHydz = -np.roll(Hy, 1, axis=2) + Hy
    else:
        dHzdy = -np.concatenate((Hz[:,-1:,:]*np.exp(1j*bloch_vector[1]*dL*1e9*Hz.shape[1]), Hz[:,:-1,:]), axis=1) + Hz
        dHydz = -np.concatenate((Hy[:,:,-1:]*np.exp(1j*bloch_vector[2]*dL*1e9*Hy.shape[2]), Hy[:,:,:-1]), axis=2) + Hy

    Ex = (dHzdy - dHydz) / (1j*dL*omega*EPSILON_0*eps_grid)
    return Ex

def H_to_Ey(Hz, Hx, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    # dHxdz = Hx[1:,:,1:]-Hx[1:,:,0:-1]
    # dHzdx = Hz[1:,:,1:]-Hz[0:-1,:,1:]
    # Ey = (dHxdz - dHzdx) / (1j*dL*omega*EPSILON_0*eps_grid[1:,:,1:])
    # return Ey
    if bloch_vector is None:
        dHxdz = -np.roll(Hx, 1, axis=2) + Hx
        dHzdx = -np.roll(Hz, 1, axis=0) + Hz
    else:
        dHxdz = -np.concatenate((Hx[:,:,-1:]*np.exp(1j*bloch_vector[2]*dL*1e9*Hx.shape[2]), Hx[:,:,:-1]), axis=2) + Hx
        dHzdx = -np.concatenate((Hz[-1:,:,:]*np.exp(1j*bloch_vector[0]*dL*1e9*Hz.shape[0]), Hz[:-1,:,:]), axis=0) + Hz

    Ey = (dHxdz - dHzdx) / (1j*dL*omega*EPSILON_0*eps_grid)
    return Ey

def H_to_Ez(Hx, Hy, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    # dHydx = Hy[1:,1:,:]-Hy[0:-1,1:,:]
    # dHxdy = Hx[1:,1:,:]-Hx[1:,0:-1,:]
    # Ez = (dHydx - dHxdy) / (1j*dL*omega*EPSILON_0*eps_grid[1:,1:,:])
    # return Ez
    if bloch_vector is None:
        dHydx = -np.roll(Hy, 1, axis=0) + Hy
        dHxdy = -np.roll(Hx, 1, axis=1) + Hx
    else:
        dHydx = -np.concatenate((Hy[-1:,:,:]*np.exp(1j*bloch_vector[0]*dL*1e9*Hy.shape[0]), Hy[:-1,:,:]), axis=0) + Hy
        dHxdy = -np.concatenate((Hx[:,-1:,:]*np.exp(1j*bloch_vector[1]*dL*1e9*Hx.shape[1]), Hx[:,:-1,:]), axis=1) + Hx

    Ez = (dHydx - dHxdy) / (1j*dL*omega*EPSILON_0*eps_grid)
    return Ez
##########################################


########## physics, batched GPU ##########

def E_to_H_batched_GPU(Ex, Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=None):
    Hx = E_to_Hx_batched_GPU(Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=bloch_vector)
    Hy = E_to_Hy_batched_GPU(Ez, Ex, dL, omega, MU_0 = MU_0, bloch_vector=bloch_vector)
    Hz = E_to_Hz_batched_GPU(Ex, Ey, dL, omega, MU_0 = MU_0, bloch_vector=bloch_vector)
    return (Hx, Hy, Hz)

def E_to_Hx_batched_GPU(Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=None):
    if bloch_vector is None:
        dEzdy = torch.roll(Ez, -1, dims=2) - Ez
        dEydz = torch.roll(Ey, -1, dims=3) - Ey
    else:
        dEzdy = torch.cat((Ez[:,:,1:,:], Ez[:,:,0:1,:]*torch.exp(-1j*bloch_vector[1]*dL*1e9*Ez.shape[2])), dim=2) - Ez
        dEydz = torch.cat((Ey[:,:,:,1:], Ey[:,:,:,0:1]*torch.exp(-1j*bloch_vector[2]*dL*1e9*Ey.shape[3])), dim=3) - Ey

    Hx = (dEzdy - dEydz) / (-1j*dL*omega*MU_0)
    return Hx

def E_to_Hy_batched_GPU(Ez, Ex, dL, omega, MU_0 = MU_0, bloch_vector=None):
    if bloch_vector is None:
        dExdz = torch.roll(Ex, -1, dims=3) - Ex
        dEzdx = torch.roll(Ez, -1, dims=1) - Ez
    else:
        dExdz = torch.cat((Ex[:,:,:,1:], Ex[:,:,:,0:1]*torch.exp(-1j*bloch_vector[2]*dL*1e9*Ex.shape[3])), dim=2) - Ex
        dEzdx = torch.cat((Ez[:,1:,:,:], Ez[:,0:1,:,:]*torch.exp(-1j*bloch_vector[0]*dL*1e9*Ez.shape[1])), dim=1) - Ez

    Hy = (dExdz - dEzdx) / (-1j*dL*omega*MU_0)
    return Hy

def E_to_Hz_batched_GPU(Ex, Ey, dL, omega, MU_0 = MU_0, bloch_vector=None):
    if bloch_vector is None:
        dEydx = torch.roll(Ey, -1, dims=1) - Ey
        dExdy = torch.roll(Ex, -1, dims=2) - Ex
    else:
        dEydx = torch.cat((Ey[:,1:,:,:], Ey[:,0:1,:,:]*torch.exp(-1j*bloch_vector[0]*dL*1e9*Ey.shape[1])), dim=1) - Ey
        dExdy = torch.cat((Ex[:,:,1:,:], Ex[:,:,0:1,:]*torch.exp(-1j*bloch_vector[1]*dL*1e9*Ex.shape[2])), dim=2) - Ex

    Hz = (dEydx - dExdy) / (-1j*dL*omega*MU_0)
    return Hz

def H_to_E_batched_GPU(Hx, Hy, Hz, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    Ex = H_to_Ex_batched_GPU(Hy, Hz, dL, omega, eps_grid[0], EPSILON_0 = EPSILON_0, bloch_vector=bloch_vector)
    Ey = H_to_Ey_batched_GPU(Hz, Hx, dL, omega, eps_grid[1], EPSILON_0 = EPSILON_0, bloch_vector=bloch_vector)
    Ez = H_to_Ez_batched_GPU(Hx, Hy, dL, omega, eps_grid[2], EPSILON_0 = EPSILON_0, bloch_vector=bloch_vector)
    return (Ex, Ey, Ez)

def H_to_Ex_batched_GPU(Hy, Hz, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    if bloch_vector is None:
        dHzdy = -torch.roll(Hz, 1, dims=2) + Hz
        dHydz = -torch.roll(Hy, 1, dims=3) + Hy
    else:
        dHzdy = -torch.cat((Hz[:,:,-1:,:]*torch.exp(1j*bloch_vector[1]*dL*1e9*Hz.shape[2]), Hz[:,:,:-1,:]), dim=2) + Hz
        dHydz = -torch.cat((Hy[:,:,:,-1:]*torch.exp(1j*bloch_vector[2]*dL*1e9*Hy.shape[3]), Hy[:,:,:,:-1]), dim=3) + Hy

    Ex = (dHzdy - dHydz) / (1j*dL*omega*EPSILON_0*eps_grid)
    return Ex

def H_to_Ey_batched_GPU(Hz, Hx, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    if bloch_vector is None:
        dHxdz = -torch.roll(Hx, 1, dims=3) + Hx
        dHzdx = -torch.roll(Hz, 1, dims=1) + Hz
    else:
        dHxdz = -torch.cat((Hx[:,:,:,-1:]*torch.exp(1j*bloch_vector[2]*dL*1e9*Hx.shape[3]), Hx[:,:,:,:-1]), dim=2) + Hx
        dHzdx = -torch.cat((Hz[:,-1:,:,:]*torch.exp(1j*bloch_vector[0]*dL*1e9*Hz.shape[1]), Hz[:,:-1,:,:]), dim=0) + Hz

    Ey = (dHxdz - dHzdx) / (1j*dL*omega*EPSILON_0*eps_grid)
    return Ey

def H_to_Ez_batched_GPU(Hx, Hy, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    if bloch_vector is None:
        dHydx = -torch.roll(Hy, 1, dims=1) + Hy
        dHxdy = -torch.roll(Hx, 1, dims=2) + Hx
    else:
        dHydx = -torch.cat((Hy[:,-1:,:,:]*torch.exp(1j*bloch_vector[0]*dL*1e9*Hy.shape[1]), Hy[:,:-1,:,:]), dim=1) + Hy
        dHxdy = -torch.cat((Hx[:,:,-1:,:]*torch.exp(1j*bloch_vector[1]*dL*1e9*Hx.shape[2]), Hx[:,:,:-1,:]), dim=2) + Hx

    Ez = (dHydx - dHxdy) / (1j*dL*omega*EPSILON_0*eps_grid)
    return Ez

def E_to_bc_batched_GPU(Ex, Ey, Ez, dL, wl, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None):
    yeex, yeey, yeez = eps_grid
    bc_x_p = torch.cat([torch.view_as_real((f[:, -1,: ,: ]-f[:, -2,: ,: ])+1/2*(f[:, -1,: ,: ]+f[:, -2,: ,: ])*1j*2*np.pi*torch.sqrt(yeex[:, -1,: ,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)
    bc_x_n = torch.cat([torch.view_as_real((f[:, 0 ,: ,: ]-f[:, 1 ,: ,: ])+1/2*(f[:, 0 ,: ,: ]+f[:, 1 ,: ,: ])*1j*2*np.pi*torch.sqrt(yeex[:, 1 ,: ,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)
    bc_y_p = torch.cat([torch.view_as_real((f[:, : ,-1,: ]-f[:, : ,-2,: ])+1/2*(f[:, : ,-1,: ]+f[:, : ,-2,: ])*1j*2*np.pi*torch.sqrt(yeey[:, : ,-1,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)
    bc_y_n = torch.cat([torch.view_as_real((f[:, : ,0 ,: ]-f[:, : ,1 ,: ])+1/2*(f[:, : ,0 ,: ]+f[:, : ,1 ,: ])*1j*2*np.pi*torch.sqrt(yeey[:, : ,1 ,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)
    bc_z_p = torch.cat([torch.view_as_real((f[:, : ,: ,-1]-f[:, : ,: ,-2])+1/2*(f[:, : ,: ,-1]+f[:, : ,: ,-2])*1j*2*np.pi*torch.sqrt(yeez[:, : ,: ,-1])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)
    bc_z_n = torch.cat([torch.view_as_real((f[:, : ,: ,0 ]-f[:, : ,: ,1 ])+1/2*(f[:, : ,: ,0 ]+f[:, : ,: ,1 ])*1j*2*np.pi*torch.sqrt(yeez[:, : ,: ,1 ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)
    bcs = torch.stack((bc_x_p, bc_x_n, bc_y_p, bc_y_n, bc_z_p, bc_z_n), dim = -2)

    return bcs

#########################################
