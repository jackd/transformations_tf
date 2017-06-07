import tensorflow as tf
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

from transformations_tf import batch_matmul


# X = [2, 3, 5]
# n = 6
# m = 7
# Y = [4]
#
# A = np.random.random(*(X + [n, m]))
# B = np.random.random(*)

A = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
A = np.stack([A, 10*A], axis=0)
B = np.array([0, 0, 1], dtype=np.float32).reshape(3, 1)
B = np.stack([B, 3*B], axis=0)
C = np.stack([np.matmul(Ai, Bi) for Ai, Bi in zip(A, B)], axis=0)

graph = tf.Graph()
with graph.as_default():
    At = tf.constant(A)
    Bt = tf.constant(B)
    Ct = batch_matmul(At, Bt)


with tf.Session(graph=graph) as sess:
    Cv = sess.run(Ct)

print(np.allclose(Cv, C))
