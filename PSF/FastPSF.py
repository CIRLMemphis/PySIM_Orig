# -*- coding: utf-8 -*-
"""Fast Gibson-Lanni PSF solver in Python.ipynb
Adapted from 
http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import sys
# %pylab inline
import scipy.special
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from math import *
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'viridis'

print('Python {}\n'.format(sys.version))
print('NumPy\t\t{}'.format(np.__version__))
print('matplotlib\t{}'.format(matplotlib.__version__))
print('SciPy\t\t{}'.format(scipy.__version__))

# Image properties
# Size of the PSF array, pixels
size_x = 256
size_y = 256
size_z = 128

# Precision control
num_basis    = 100  # Number of rescaled Bessels that approximate the phase function
num_samples  = 1000 # Number of pupil samples along radial direction
oversampling = 1    # Defines the upsampling ratio on the image space grid for computations

# Microscope parameters
NA          = 1.4
wavelength  = 0.610 # microns
M           = 100   # magnification
ns          = 1.33  # specimen refractive index (RI)
ng0         = 1.5   # coverslip RI design value
ng          = 1.5   # coverslip RI experimental value
ni0         = 1.5   # immersion medium RI design value
ni          = 1.5   # immersion medium RI experimental value
ti0         = 150   # microns, working distance (immersion medium thickness) design value
tg0         = 170   # microns, coverslip thickness design value
tg          = 170   # microns, coverslip thickness experimental value
res_lateral = 0.1   # microns
res_axial   = 0.25  # microns
pZ          = 0     # microns, particle distance from coverslip
#pZ          = 2     # microns, particle distance from coverslip

# Scaling factors for the Fourier-Bessel series expansion
min_wavelength = 0.436 # microns
scaling_factor = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wavelength / wavelength

# Place the origin at the center of the final PSF array
x0 = (size_x - 1) / 2
y0 = (size_y - 1) / 2

# Find the maximum possible radius coordinate of the PSF array by finding the distance
# from the center of the array to a corner
max_radius = round(sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0))) + 1;

# Radial coordinates, image space
r = res_lateral * np.arange(0, oversampling * max_radius) / oversampling

# Radial coordinates, pupil space
a = min([NA, ns, ni, ni0, ng, ng0]) / NA
rho = np.linspace(0, a, num_samples)

# Stage displacements away from best focus
#z = res_axial * np.arange(-size_z / 2, size_z /2) + res_axial / 2
z = res_axial * np.arange(-size_z / 2, size_z /2)

# Define the wavefront aberration
ts   = -5.0
OPDs = ts * np.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample
ti   = ni * (-tg/ng + tg0/ng0 + ti0/ni0 - ts/ns)
OPDi = (z.reshape(-1,1) + pZ + ti) * np.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * np.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium
OPDg = tg * np.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * np.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip
W    = 2 * np.pi / wavelength * (OPDs + OPDi + OPDg)

# Sample the phase
# Shape is (number of z samples by number of rho samples)
phase = np.cos(W) + 1j * np.sin(W)

# Define the basis of Bessel functions
# Shape is (number of basis functions by number of rho samples)
J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

# Compute the approximation to the sampled pupil phase by finding the least squares
# solution to the complex coefficients of the Fourier-Bessel expansion.
# Shape of C is (number of basis functions by number of z samples).
# Note the matrix transposes to get the dimensions correct.
C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T, rcond=-1)

print(np.shape(J))
print(np.shape(phase))
print(np.shape(C))

# Which z-plane to compute
z0 = 24

# The Fourier-Bessel approximation
est = J.T.dot(C[:,z0])

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10,5))
ax[0].plot(rho, np.real(phase[z0, :]), label=r'$ \exp{ \left[ jW \left( \rho \right) \right] }$')
ax[0].plot(rho, np.real(est), '--', label='Fourier-Bessel')
ax[0].set_xlabel(r'$\rho$')
ax[0].set_title('Real')
ax[0].legend(loc='upper left')

ax[1].plot(rho, np.imag(phase[z0, :]))
ax[1].plot(rho, np.imag(est), '--')
ax[1].set_xlabel(r'$\rho$')
ax[1].set_title('Imaginary')
ax[1].set_ylim((-1.5, 1.5))

plt.show()

b = 2 * np. pi * r.reshape(-1, 1) * NA / wavelength

# Convenience functions for J0 and J1 Bessel functions
J0 = lambda x: scipy.special.jv(0, x)
J1 = lambda x: scipy.special.jv(1, x)

# See equation 5 in Li, Xue, and Blu
denom = scaling_factor * scaling_factor - b * b
R = (scaling_factor * J1(scaling_factor * a) * J0(b * a) * a - b * J0(scaling_factor * a) * J1(b * a) * a)
R /= denom

# The transpose places the axial direction along the first dimension of the array, i.e. rows
# This is only for convenience.
PSF_rz = (np.abs(R.dot(C))**2).T

# Normalize to the maximum value
PSF_rz /= np.max(PSF_rz)

fig, ax = plt.subplots()

ax.imshow(PSF_rz, extent=(r.min(), r.max(), z.max(), z.min()))
ax.set_xlim((0,2.5))
ax.set_ylim((-6, 0))
ax.set_xlabel(r'r, $\mu m$')
ax.set_ylabel(r'z, $\mu m$')

plt.show()

# Create the fleshed-out xy grid of radial distances from the center
xy      = np.mgrid[0:size_y, 0:size_x]
r_pixel = np.sqrt((xy[1] - x0) * (xy[1] - x0) + (xy[0] - y0) * (xy[0] - y0)) * res_lateral

PSF = np.zeros((size_y, size_x, size_z))

for z_index in range(PSF.shape[2]):
    # Interpolate the radial PSF function
    PSF_interp = interp1d(r, PSF_rz[z_index, :])
    
    # Evaluate the PSF at each value of r_pixel
    PSF[:,:, z_index] = PSF_interp(r_pixel.ravel()).reshape(size_y, size_x)

plt.style.use('classic')
plt.figure(figsize=(15, 9.5))
plt.subplot(1,2,1)
plt.imshow(PSF[:,:,1+int(size_z/2)], cmap='jet')
plt.subplot(1,2,2)
plt.imshow(PSF[1+int(size_x/2),:,:], cmap='jet')
plt.show()
plt.savefig('3D_PSF.png')

plt.style.use('classic')
plt.figure(figsize=(11, 5.5))
plt.plot(PSF[int(size_x/2),int(size_x/2),:])
plt.show()
plt.savefig('PSFprofile.png')
