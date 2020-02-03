# Load Tensorflow==2.0
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)

### The input to the autodiff function is a tensor of dimensions specified by you. 
# Let us specify a tensor of 2 x 2 and value of "ones"
# input tensor x, size(2,2) 
x = tf.ones((2, 2))
print ("\n x: is a 2 x 2 Tensor with Ones" , x)

### Next We use the tf.GradientTape API for automatic differentiation and gradient calculation considering our input variables.

# the tf.gradient (m, n) function will record all computations done to a "tape" and Tensorflow employs the tape and gradients associated
# with each recorded operation to compute the gradients of a "recorded" computation through reverse mode differentiation.

# Initialize "tape" as t for recording operations
with tf.GradientTape() as t:        # load tape for the operations on objects
  t.watch(x)                        # records tape object
# operation 
  y = tf.reduce_sum(x)              # y = sum along dim(tensor x) == tensor rank 0, val  = 4
  print("\ny1:", y)
  z = tf.multiply(y, y)             # z= y^2 
  print("\nz1:", z)

# To find d(z)/dx 
### We know dy/dx is still (x = tf.ones((2, 2)) and dz/dy = 2y at, y = 4.
# Hence dz/dy = dz/dy * dy/dx = 2(4) * (x = tf.ones((2, 2)) = 8 * (x = tf.ones((2, 2)) 

dz_dx = t.gradient(z, x)            # dz/dx = dz/dy * dy/dx
print("\ndz_dz:", dz_dx)

# To confirm our value, Uncomment this
#for i in [0, 1]:
#  for j in [0, 1]:
#    assert dz_dx[i][j].numpy() == 8.0
