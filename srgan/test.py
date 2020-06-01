import tensorflow as tf
import itertools
from mat_file import mat_file

mf = mat_file()

def gen():
	X_train, X_test, y_train, y_test = mf.get_images()
	print(len(X_train))
	for i in range(len(X_train)):
		yield (X_train[i], X_train[i])

dataset = tf.data.Dataset.from_generator(gen,(tf.float32, tf.float32))

x = list(dataset.take(3))
#print(x)
