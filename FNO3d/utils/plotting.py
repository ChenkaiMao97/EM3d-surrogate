import numpy as np
from matplotlib import pyplot as plt

import matplotlib.cm as cm
from matplotlib.colors import Normalize

def plot_3slices(data, fname, stride = 1, my_cm = plt.cm.binary, cm_zero_center=True):
    # using3D()
    sx, sy, sz = data.shape
    xy_slice = data[:, :, int(sz/2)]
    yz_slice = data[int(sx/2), :, :]
    zx_slice = data[:, int(sy/2), :].T

    x = list(range(sx))
    y = list(range(sy))
    z = list(range(sz))

    fig = plt.figure(figsize=(14,4))
    ax = plt.subplot(131, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x1 = np.array([0*i + j for j in x for i in y]).reshape((sx,sy))
    y1 = np.array([i + 0*j for j in x for i in y]).reshape((sx,sy))
    z1 = sz/2*np.ones((len(x), len(y)))
    if cm_zero_center:
        vm = max(np.max(xy_slice), -np.min(xy_slice))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(xy_slice), vmax=np.max(xy_slice))

    # plt.figure()
    # plt.imshow(xy_slice)
    # plt.savefig("debug.png")
    # plt.close()

    surf = ax.plot_surface(x1.T, y1.T, z1.T, rstride=stride, cstride=stride, facecolors=my_cm(norm(xy_slice.T)))
    ax.set_zlim((0,sz))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cm)
    mappable.set_array(xy_slice)
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)


    ax = plt.subplot(132, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x2 = sx/2*np.ones((len(y), len(z)))
    y2 = np.array([0*i + j for j in y for i in z]).reshape((sy,sz))
    z2 = np.array([i + 0*j for j in y for i in z]).reshape((sy,sz))
    if cm_zero_center:
        vm = max(np.max(yz_slice), -np.min(yz_slice))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(yz_slice), vmax=np.max(yz_slice))
    ax.plot_surface(x2, y2, z2, rstride=stride,cstride=stride, facecolors=my_cm(norm(yz_slice)))
    ax.set_xlim((0,sx))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cm)
    mappable.set_array(yz_slice)
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    
    ax = plt.subplot(133, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x3 = np.array([i + 0*j for j in z for i in x]).reshape((sz,sx))
    y3 = sy/2*np.ones((len(z), len(x)))
    z3 = np.array([0*i + j for j in z for i in x]).reshape((sz,sx))
    if cm_zero_center:
        vm = max(np.max(zx_slice), -np.min(zx_slice))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(zx_slice), vmax=np.max(zx_slice))
    ax.plot_surface(x3, y3, z3, rstride=stride,cstride=stride, facecolors=my_cm(norm(zx_slice)))
    ax.set_ylim((0,sy))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cm)
    mappable.set_array(zx_slice)
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(fname, dpi=500, transparent=True)
    plt.close()

    # example:
    # eps = np.load(folder+"epsilon_DDM.npy").astype(np.float32)
    # print(f"eps: {eps.shape}, {eps.dtype}, max: {np.max(eps)}, min: {np.min(eps)}, mean: {np.mean(eps)}")
    # fig = plot_3slices(eps[0], cm_zero_center=False)
    # plt.tight_layout()
    # plt.savefig("eps.png", dpi=500, transparent=True)

    # Ex = np.real(np.load(folder+str(error_th)+"Ex.npy").astype(np.complex64))
    # print(f"Ex: {Ex.shape}, {Ex.dtype}")
    # fig = plot_3slices(Ex, my_cm=plt.cm.seismic)
    # plt.tight_layout()
    # plt.savefig("Ex_real.png", dpi=500, transparent=True)

