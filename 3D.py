# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:40:27 2026

@author: mjorg
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt

# =====================================
# GPU
# =====================================
device = "cuda"

# =====================================
# parámetros físicos
# =====================================
dt = 0.001
N = 50000

gamma = 1.0

# rigideces de la trampa
kx = 0.8
ky = 0.6
kz = 0.3

# difusión térmica
D = 0.4

# =====================================
# tensores GPU
# =====================================
x = torch.zeros(N, device=device)
y = torch.zeros(N, device=device)
z = torch.zeros(N, device=device)

# condición inicial
x[0] = 1.5
y[0] = -1.0
z[0] = 0.8

# amplitud del ruido
noise_scale = torch.sqrt(torch.tensor(2 * D * dt, device=device))

# =====================================
# simulación
# =====================================
for i in range(N - 1):

    # ruido browniano
    nx = noise_scale * torch.randn(1, device=device)
    ny = noise_scale * torch.randn(1, device=device)
    nz = noise_scale * torch.randn(1, device=device)

    # ecuaciones de Langevin
    dx = (-kx / gamma) * x[i] * dt + nx
    dy = (-ky / gamma) * y[i] * dt + ny
    dz = (-kz / gamma) * z[i] * dt + nz

    # actualizar posiciones
    x[i + 1] = x[i] + dx
    y[i + 1] = y[i] + dy
    z[i + 1] = z[i] + dz

# =====================================
# mover a CPU
# =====================================
x_cpu = x.cpu().numpy()
y_cpu = y.cpu().numpy()
z_cpu = z.cpu().numpy()

# =====================================
# gráfica 3D
# =====================================
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')

ax.plot(x_cpu, y_cpu, z_cpu, linewidth=0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Trayectoria Browniana 3D en trampa óptica")

plt.show()

# =====================================
# información CUDA
# =====================================
print("CUDA disponible:", torch.cuda.is_available())
print("GPU usada:", torch.cuda.get_device_name(0))