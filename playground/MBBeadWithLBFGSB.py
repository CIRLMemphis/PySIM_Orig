# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import tensorflow_probability as tfp
tf.__version__
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

print(tf.test.is_gpu_available())

"""Load the forward images"""

annots = loadmat('data//SimTunableA5BhuBead.mat')
g = annots['g']
print(np.shape(g))
g = np.transpose(g, (3, 4, 2, 0, 1)) # permute the array to [Nthe, Nphi, Z, X, Y]
Nthe, Nphi, Z, X, Y = np.shape(g)

"""Load the PSF and the patterns"""

res = loadmat('data//hBhuBead.mat')
h   = res['h']
h   = np.transpose(h, (2, 0, 1))  # permute the array to [Z, X, Y]
res = loadmat('data//imBhuBead.mat')
im  = res['im']
im  = np.transpose(im, (5, 3, 4, 2, 0, 1)) # permute the array to [Nm, Nthe, Nphi, Z, X, Y]
res = loadmat('data//jmBhuBead.mat')
jm  = res['jm']
jm  = np.transpose(jm, (5, 3, 4, 2, 0, 1)) # permute the array to [Nm, Nthe, Nphi, Z, X, Y]

def flip_from_dim(data, dim):
    for i in range(dim, len(data.shape)):
        data = np.flip(data,i) #Reverse given tensor on dimention i
    return data

#h  = flip_from_dim(h , 0) # flip because tf.conv3d do cross-correlation instead of convolution
#im = flip_from_dim(im, 3) # flip because tf.conv3d do cross-correlation instead of convolution

guess = np.zeros((Z,X,Y))
for l in range(Nthe):
  for k in range(Nphi):
    guess = guess + g[l,k,:,:,:]
guess = guess/(Nthe*Nphi)
plt.style.use('classic')
plt.figure(figsize=(8, 3.5))
plt.subplot(1,2,1)
plt.imshow(guess[150,:,:], cmap='gray')
plt.subplot(1,2,2)
plt.imshow(guess[:,100,:], cmap='gray')
plt.show()
plt.savefig('guess.png')

hTf  = tf.constant(h , dtype=tf.complex64) 
imTf = tf.constant(im, dtype=tf.complex64)
jmTf = tf.constant(jm, dtype=tf.complex64)
gTf  = tf.convert_to_tensor(g, dtype=tf.float32)

def costFct(m):
  mRe     = tf.dtypes.cast(tf.reshape(m, [300, 200, 200]), dtype=tf.complex64)
  curCost = 0.0;
  for l in range(Nthe):
    for k in range(Nphi):
      objm = mRe*jmTf[1,l,k,:,:,:]
      him  = hTf*imTf[1,l,k,:,:,:]
      G    = tf.signal.fft3d(mRe)*tf.signal.fft3d(hTf) + tf.signal.fft3d(objm)*tf.signal.fft3d(him)
      gCur = tf.math.real(tf.signal.ifftshift(tf.signal.ifft3d(G)))
      curCost = curCost + tf.reduce_sum(tf.square(gCur - gTf[l,k,:,:,:]))
  print(curCost)
  return curCost

start = tf.reshape(tf.Variable(initial_value=guess, shape=(300,200,200), dtype=tf.float32), [300*200*200])
optim_results = tfp.optimizer.lbfgs_minimize(
    lambda m: tfp.math.value_and_gradient(costFct, m),
    initial_position=start,
    num_correction_pairs=10,
    max_iterations=100,
    tolerance=1e-8,
)

reconOb = np.reshape(optim_results.position.numpy(), (300,200,200))
plt.style.use('classic')
plt.figure(figsize=(8, 3.5))
plt.subplot(1,2,1)
plt.imshow(reconOb[150,:,:], cmap='gray')
plt.subplot(1,2,2)
plt.imshow(reconOb[:,100,:], cmap='gray')
plt.show()
plt.savefig('reconOb.png')
