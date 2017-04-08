"""Test cases for various transformation functions."""

import tensorflow as tf
import numpy as np

import transformations_tf as ttf
import transformations as t

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

# is_close
# all_close
# identity_matrix
# translation_from_matrix
# reflection_matrix
# rotation_matrix
# homogeneous_rotation_matrix
# euler_matrix_nh
# euler_matrix
# quaternion_from_euler
# quaternion_about_axis
# quaternion_matrix_nh
# quaternion_matrix
# quaternion_multiply
# quaternion_conjugate
# quaternion_inverse
# quaternion_real
# quaternion_imag
# vector_norm2
# vector_norm
# unit_vector
# outer
# vector_product
# angle_between_vectors
# inverse_matrix
# concatenate_matrices
# is_same_transform
# is_same_quaternion


def _random_quaternion(dtype):
    return t.random_quaternion().astype(dtype)


def _listwrap(value):
    if isinstance(value, (float, int, np.ndarray)):
        value = value,
    assert(isinstance(value, (list, tuple)))
    return value


class TransformationsTestCpu(test_util.TensorFlowTestCase):
    """
    Class for testing `transformations_tf` functions.

    Tests only run for cpu implementations.
    """

    _use_gpu = False
    _batch_len = 6

    def _np_tf_compare(
            self, input_fn, tf_fn, np_fn,
            dtypes=[tf.float32, tf.float64]):
        """Compare numpy and tensorflow implementations."""
        with self.test_session(use_gpu=self._use_gpu) as sess:
            for dtype in dtypes:
                np_dtype = dtype.as_numpy_dtype

                in_np = [_listwrap(input_fn(np_dtype))
                         for i in range(self._batch_len)]
                out_np = [_listwrap(np_fn(*inp)) for inp in in_np]

                in_np = [np.array(z) for z in zip(*in_np)]
                out_np = [np.array(z) for z in zip(*out_np)]

                in_tf = [tf.constant(i, dtype=dtype) for i in in_np]
                out_tf = _listwrap(sess.run(tf_fn(*in_tf)))

                for o_tf, o_np in zip(out_tf, out_np):
                    if len(o_tf.shape) != len(o_np.shape):
                        for o_npi in o_np:
                            self.assertAllClose(o_tf, o_npi)
                    else:
                        self.assertAllClose(o_tf, o_np)

    def test_close(self):
        """Test is_close, all_close."""
        shape = (5, 4, 7)

        def input_fn(dtype):
            x = np.random.random(shape).astype(dtype)
            y = (np.random.random(shape)*1e-6).astype(dtype)
            return x, x + y

        def tf_fn(x, z):
            return ttf.is_close(x, z), \
                ttf.all_close(x, z, axis=tuple(range(1, len(shape)+1)))

        def np_fn(x, z):
            return np.isclose(x, z), np.allclose(x, z)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_outer(self):
        """Test tensorflow implementation of outer."""
        def input_fn(dtype):
            return [np.random.random(3).astype(dtype) for i in range(2)]

        self._np_tf_compare(input_fn, ttf.outer, np.outer)

    def test_vector_product(self):
        """Test tensorflow implementation of vector_product."""
        def input_fn(dtype):
            return [np.random.random(3).astype(dtype) for i in range(2)]

        self._np_tf_compare(input_fn, ttf.vector_product, t.vector_product)

    def test_angle_between_vectors(self):
        """Test tensorflow implementation of angle_between_vectors."""
        def input_fn(dtype):
            return [np.random.random(3).astype(dtype) for i in range(2)]

        self._np_tf_compare(
            input_fn,
            ttf.angle_between_vectors,
            t.angle_between_vectors)

    def test_identity_matrix(self):
        """Test identity_matrix function."""
        def input_fn(dtype):
            return t.identity_matrix().astype(dtype)

        def tf_fn(in_tf):
            return ttf.identity_matrix(dtype=in_tf.dtype)

        def np_fn(in_np):
            return in_np

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_translation_from_matrix(self):
        """Test tensorflow implementation of translation_from_matrix."""
        def input_fn(dtype):
            return np.random.random((4, 4)).astype(dtype)

        def tf_fn(m):
            return ttf.translation_from_matrix(m)

        def np_fn(m):
            return t.translation_from_matrix(m)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_reflection_matrix(self):
        """Test tensorflow implementation of reflection_matrix."""
        def input_fn(dtype):
            return np.random.random((4, 4)).astype(dtype)

        def tf_fn(m):
            return ttf.translation_from_matrix(m)

        def np_fn(m):
            return t.translation_from_matrix(m)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_homogeneous_rotation_matrix(self):
        """Test tensorflow implementation of homogeneous_rotation_matrix."""
        def input_fn(dtype):
            m = t.random_rotation_matrix().astype(dtype)
            R = m[:3, :3]
            return m, R

        def tf_fn(m, R):
            return ttf.homogeneous_rotation_matrix(R)

        def np_fn(m, R):
            return m

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_rotation_matrix_nh(self):
        """Test tensorflow implementation of rotation_matrix_nh."""
        def input_fn(dtype):
            angle = np.array(np.random.random()*2*np.pi, dtype=dtype)
            axis = np.random.random(3).astype(dtype)
            return angle, axis

        def tf_fn(angle, axis):
            r = ttf.rotation_matrix_nh(angle, axis)
            return r

        def np_fn(angle, axis):
            return t.rotation_matrix(angle, axis)[:3, :3].astype(angle.dtype)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_rotation_matrix(self):
        """Test tensorflow implementation of rotation_matrix."""
        def input_fn(dtype):
            angle = np.array(np.random.random()*2*np.pi, dtype=dtype)
            axis = np.random.random(3).astype(dtype)
            return angle, axis

        def tf_fn(angle, axis):
            return ttf.rotation_matrix(angle, axis)

        def np_fn(angle, axis):
            return t.rotation_matrix(angle, axis).astype(angle.dtype)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_euler_matrix_nh(self):
        """Test tensorflow implementation of euler_matrix_nh."""
        def input_fn(dtype):
            return (np.random.random(3)*2*np.pi).astype(dtype)

        def tf_fn(angles):
            return ttf.euler_matrix_nh(*[angles[..., i] for i in range(3)])

        def np_fn(angles):
            return t.euler_matrix(*angles)[:3, :3].astype(angles.dtype)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_euler_matrix(self):
        """Test tensorflow implementation of euler_matrix."""
        def input_fn(dtype):
            return (np.random.random(3)*2*np.pi).astype(dtype)

        def tf_fn(angles):
            return ttf.euler_matrix(*[angles[..., i] for i in range(3)])

        def np_fn(angles):
            return t.euler_matrix(*angles).astype(angles.dtype)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_quaternion_from_euler(self):
        """Test tensorflow implementation of quaternion_from_euler."""
        def input_fn(dtype):
            return (np.random.random(3)*2*np.pi).astype(dtype)

        def tf_fn(angles):
            return ttf.quaternion_from_euler(
                *[angles[..., i] for i in range(3)])

        def np_fn(angles):
            return t.quaternion_from_euler(*angles).astype(angles.dtype)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_quaternion_about_axis(self):
        """Test tensorflow implementation of quaternion_about_axis."""
        def input_fn(dtype):
            angle = np.array(np.random.random()*2*np.pi, dtype=dtype)
            axis = np.random.random(3).astype(dtype)
            return angle, axis

        def tf_fn(angle, axis):
            return ttf.quaternion_about_axis(angle, axis)

        def np_fn(angle, axis):
            return t.quaternion_about_axis(angle, axis)

        self._np_tf_compare(input_fn, tf_fn, np_fn)

    def test_quaternion_matrix_nh(self):
        """Test tensorflow implementation of quaternion_matrix_nh."""
        def np_fn(q):
            return t.quaternion_matrix(q)[:3, :3].astype(q.dtype)

        self._np_tf_compare(
            _random_quaternion,
            ttf.quaternion_matrix_nh,
            np_fn)

    def test_quaternion_matrix(self):
        """Test tensorflow implementation of quaternion_matrix."""
        self._np_tf_compare(
            _random_quaternion,
            ttf.quaternion_matrix,
            t.quaternion_matrix)

    def test_quaternion_multiply(self):
        """Test tensorflow implementation of quaternion_multiply."""
        def input_fn(dtype):
            return [_random_quaternion(dtype) for i in range(2)]

        self._np_tf_compare(
            input_fn, ttf.quaternion_multiply, t.quaternion_multiply)

    def test_quaternion_conjugate(self):
        """Test tensorflow implementation of quaternion_conjugate."""
        self._np_tf_compare(
            _random_quaternion,
            ttf.quaternion_conjugate,
            t.quaternion_conjugate)

    def test_quaternion_inverse(self):
        """Test tensorflow implementation of quaternion_inverse."""
        self._np_tf_compare(
            _random_quaternion,
            ttf.quaternion_inverse,
            t.quaternion_inverse)

    def test_quaternion_real(self):
        """Test tensorflow implementation of quaternion_real."""
        self._np_tf_compare(
            _random_quaternion,
            ttf.quaternion_real,
            t.quaternion_real)

    def test_quaternion_imag(self):
        """Test tensorflow implementation of quaternion_imag."""
        self._np_tf_compare(
            _random_quaternion,
            ttf.quaternion_imag,
            t.quaternion_imag)

    def test_vector_norm2(self):
        """Test tensorflow implementation of vector_norm2."""
        shape = (5, 4, 7)

        def input_fn(dtype):
            return np.random.random(shape).astype(dtype)

        def np_fn(v):
            return np.sum(v**2, axis=-1)

        self._np_tf_compare(input_fn, ttf.vector_norm2, np_fn)

    def test_vector_norm(self):
        """Test tensorflow implementation of vector_norm."""
        shape = (5, 4, 7)

        def input_fn(dtype):
            return np.random.random(shape).astype(dtype)

        def np_fn(v):
            return np.sqrt(np.sum(v**2, axis=-1))

        self._np_tf_compare(input_fn, ttf.vector_norm, np_fn)

    def test_unit_vector(self):
        """Test tensorflow implementation of unit_vector."""
        def input_fn(dtype):
            return np.random.random(3).astype(dtype)

        self._np_tf_compare(input_fn, ttf.unit_vector, t.unit_vector)

    def test_inverse_matrix(self):
        """Test tensorflow implementation of inverse_matrix."""
        def input_fn(dtype):
            return np.random.random((5, 5)).astype(dtype)

        self._np_tf_compare(input_fn, ttf.inverse_matrix, t.inverse_matrix)

    def test_concatenate_matrices(self):
        """Test tensorflow implementation of concatenate_matrices."""
        def input_fn(dtype):
            return tuple(np.random.random((4, 4)) for i in range(5))

        self._np_tf_compare(
            input_fn,
            ttf.concatenate_matrices,
            t.concatenate_matrices)

    def test_is_same_transform(self):
        """Test tensorflow implementation of is_same_transform."""
        def input_fn_true(dtype):
            m0 = np.random.random((4, 4)).astype(dtype)
            m1 = 5*m0
            return m0, m1

        def input_fn_false(dtype):
            m0 = np.random.random((4, 4)).astype(dtype)
            m1 = np.random.random((4, 4)).astype(dtype)
            return m0, m1

        for input_fn in [input_fn_true, input_fn_false]:
            self._np_tf_compare(
                input_fn, ttf.is_same_transform, t.is_same_transform)

    def test_is_same_quaternion(self):
        """Test tensorflow implementation of is_same_quaternion."""
        def input_fn_pos(dtype):
            q = _random_quaternion(dtype)
            return q, q

        def input_fn_neg(dtype):
            q = _random_quaternion(dtype)
            return q, q

        def input_fn_false(dtype):
            return [_random_quaternion(dtype) for i in range(2)]

        for input_fn in [input_fn_pos, input_fn_neg, input_fn_false]:
            self._np_tf_compare(
                input_fn, ttf.is_same_quaternion, t.is_same_quaternion)


class TransformationsTestGpu(TransformationsTestCpu):
    """
    Class for testing `transformations_tf` functions.

    Tests only run for gpu implementations.
    """

    _use_gpu = False


if __name__ == "__main__":
    googletest.main()
