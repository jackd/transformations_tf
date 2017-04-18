"""
Port of `transformations` for tensorflow.

Original (non-tf) code by
Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>

All operations are for batches.
"""

from __future__ import division, print_function
import tensorflow as tf
import numpy as np


def _eps(dtype):
    return np.finfo(dtype.as_numpy_dtype).eps


def _stack_recursive(M, axis=0):
    if axis < 0:
        Ms = M
        while isinstance(Ms, (list, tuple)):
            Ms = Ms[0]
        axis += Ms.shape.ndims + 1
    assert(axis >= 0)
    if isinstance(M, (list, tuple)):
        return tf.stack(
            [_stack_recursive(m, axis=axis) for m in M], axis=axis)
    else:
        return M


def is_close(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Tensorflow equivalent of `np.isclose`."""
    if equal_nan:
        raise NotImplementedError()
    return tf.less_equal(tf.abs(a - b), atol + rtol * tf.abs(b))


def all_close(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, axis=None):
    """Tensorflow equivalent of `np.allclose`."""
    if equal_nan:
        raise NotImplementedError()
    return tf.reduce_all(
        is_close(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan), axis=axis)


# def _ei(i, n, dtype=tf.float32, name='ei'):
#     """Get a tensor of `n` zeros except the `i`th, which is one."""
#     if not (isinstance(i, int) and isinstance(n, int)):
#         raise TypeError('both i and n must be ints.')
#     if i >= n:
#         raise ValueError(
#             'i must be less than n, but i = %d, n = %d' % (i, n))
#     z = np.zeros((n,))
#     z[i] = 1
#     return tf.constant(z, dtype=dtype, name=name)


def identity_matrix(n=4, dtype=tf.float32, name='identity'):
    """Get a 4x4 identity matrix."""
    return tf.constant(
        np.identity(n, dtype=dtype.as_numpy_dtype), name=name, dtype=dtype)


# def translation_matrix(direction):
#     """Get the matrix transform for translation by `direction`."""
#     raise NotImplementedError()


def translation_from_matrix(matrix):
    """Extract the transflation part of the transformation matrix."""
    return matrix[..., :3, 3]


def reflection_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal vector."""
    dtype = point.dtype
    if normal.dtype != point.dtype:
        raise ValueError('point and normal must be same dtype')
    if dtype not in [tf.float32, tf.float64]:
        raise ValueError('dtype not in allowable list [float32, float64]')
    normal = unit_vector(normal[..., :3])
    dims = point.shape
    ndims = dims.ndims
    i3 = tf.identity_mnatrix(3)
    z3 = tf.zeros((1,) * (ndims-1) + (3, 1), dtype=dtype)
    one = tf.ones(dims[:-1].as_list() + [1], dtype=dtype)
    M = tf.stack([
        tf.stack([-2*outer(normal, normal) + i3, z3], axis=-1),
        tf.stack([2 * tf.dot(point[..., :3], normal) * normal, one], axis=-1),
    ], axis=-1)
    return M


# def reflection_from_matrix(matrix):
#     """Get the point/normal that represent this matrix transformation."""
#     raise NotImplementedError()


def batch_matmul(A, B):
    """
    Perform matrix multiplication A*B for independent rows in a batch.

    If A.shape == X + [n, m], then B.shape == X + [m] + Y and the returned
    tensor C will satisfy C.shape == X + [n] + Y.
    """
    # X = A.shape[:-2]
    # n = A.shape[-2]
    # m = A.shape[-1]
    X = A.shape[:-2]
    nx = len(X)
    m = A.shape[-1]
    Y = B.shape[nx + 1:]

    if B.shape[nx] not in [m, 1]:
        raise IndexError()
    for i in range(len(Y)):
        A = tf.expand_dims(A, axis=-1)
    B = tf.expand_dims(B, nx)
    C = tf.reduce_sum(A*B, axis=nx+1)
    return C


def rotation_matrix_nh(angle, direction):
    """Get the nh rotation matrix about axis `direction` by angle `angle`."""
    sina = tf.sin(angle)
    cosa = tf.cos(angle)
    direction = unit_vector(direction)
    o = outer(direction, direction) * (
        1. - tf.expand_dims(tf.expand_dims(cosa, -1), -1))
    # print(o.shape)
    # raise Exception()
    direction *= tf.expand_dims(sina, -1)
    d = tf.unstack(direction, axis=-1)
    R = tf.stack([
        tf.stack([cosa, -d[2], d[1]], axis=-1),
        tf.stack([d[2], cosa, -d[0]], axis=-1),
        tf.stack([-d[1], d[0], cosa], axis=-1)
    ], axis=-2) + o
    return R


def rotation_matrix(angle, direction):
    """Get the rotation matrix about axis `direction` by angle `angle`."""
    return homogeneous_rotation_matrix(rotation_matrix_nh(angle, direction))


# def scale_matrix(factor, origin=None, direction=None):
#     """Return matrix toscale by factor around origin in direction."""
#     raise NotImplementedError()
#
#
# def scale_from_matrix(matrix):
#     """Return scaling factor, origin and direction from scaling matrix."""
#     raise NotImplementedError()
#
#
# def projection_matrix(point, normal, direction=None,
#                       perspective=None, pseudo=False):
#     """Return matrix to project onto plane defined by point and normal."""
#     raise NotImplementedError()
#
#
# def projection_frommatrix(matrix, pseudo=False):
#     """Return projection plane and perspective point from matrix."""
#     raise NotImplementedError()
#
#
# def clip_matrix(left, right, bottom, top, near, far, perspective=False):
#     """Return matrix to obtain normalized device coordinates from frustum."""
#     raise NotImplementedError()
#
#
# def shear_matrix(angle, direction, point, normal):
#     """Return matrix to shear by angle along direction vector on shear plane.
#
#     The shear plane is defined by a point and normal vector. The direction
#     vector must be orthogonal to the plane's normal vector.
#
#     A point P is transformed by the shear matrix into P" such that
#     the vector P-P" is parallel to the direction vector and its extent is
#     given by the angle of P-P'-P", where P' is the orthogonal projection
#     of P onto the shear plane.
#     """
#     raise NotImplementedError()
#
#
# def shear_from_matrix(matrix):
#     """Return shear angle, direction and plane from shear matrix."""
#     raise NotImplementedError()

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

_NEXT_AXIS = [1, 2, 0, 1]


def euler_matrix_nh(ai, aj, ak, axes='sxyz'):
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    # if ai.dtype != aj.dtype or ai.dtype != ak.dtype:
    #     raise ValueError('ai, aj and ak must all be same dtype.')
    # if ai.dtype not in [tf.float32, tf.float64]:
    #     raise ValueError('angle values must be in [float32, float64]')

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = tf.sin(ai), tf.sin(aj), tf.sin(ak)
    ci, cj, ck = tf.cos(ai), tf.cos(aj), tf.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = [[None for ii in range(3)] for jj in range(3)]
    if repetition:
        M[i][i] = cj
        M[i][j] = sj*si
        M[i][k] = sj*ci
        M[j][i] = sj*sk
        M[j][j] = -cj*ss+cc
        M[j][k] = -cj*cs-sc
        M[k][i] = -sj*ck
        M[k][j] = cj*sc+cs
        M[k][k] = cj*cc-ss
    else:
        M[i][i] = cj*ck
        M[i][j] = sj*sc-cs
        M[i][k] = sj*cc+ss
        M[j][i] = cj*sk
        M[j][j] = sj*ss+cc
        M[j][k] = sj*cs-sc
        M[k][i] = -sj
        M[k][j] = cj*si
        M[k][k] = cj*ci
    M = _stack_recursive(M, axis=-1)
    return M


def homogeneous_rotation_matrix(R):
    """
    Convert the non-homoegneous rotation matrix `R` to homogeneous matrix.

    M = [[R, [0, 0, 0]], [0, 0, 0, 1]]
    """
    if not (R.get_shape().as_list()[-2:] == [3, 3]):
        raise ValueError(
            'R.shape[-2:] != (3, 3) (not a homogeneous transform)')
    z = tf.zeros_like(R)
    z_row = z[..., :1, :]
    left = tf.concat([R, z_row], axis=-2)

    z_col = z[..., :1]
    ones = tf.ones_like(R)[..., :1, :1]
    right = tf.concat([z_col, ones], axis=-2)
    return tf.concat([left, right], axis=-1)


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    """
    return homogeneous_rotation_matrix(euler_matrix_nh(ai, aj, ak, axes=axes))


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = tf.cos(ai)
    si = tf.sin(ai)
    cj = tf.cos(aj)
    sj = tf.sin(aj)
    ck = tf.cos(ak)
    sk = tf.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = [None for ii in range(4)]
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0
    q = tf.stack(q, axis=-1)
    return q


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis."""
    axis_len = vector_norm(axis)

    def if_true():
        return axis*tf.expand_dims(tf.sin(angle/2.0) / axis_len, -1)

    def if_false():
        return axis

    axis = tf.where(
        tf.greater(axis_len, _eps(angle.dtype)), if_true(), if_false())
    q0 = tf.cos(angle/2.0)
    q = tf.concat([tf.expand_dims(q0, axis=-1), axis], axis=-1)
    return q


def quaternion_matrix_nh(q0):
    """Return nonhomogeneous rotation matrix from quaternion."""
    n = vector_norm2(q0)
    cond = tf.less(n, _eps(q0.dtype))

    def if_false():
        q = q0*tf.expand_dims(tf.sqrt(2.0 / n), -1)
        q = outer(q, q)
        q = [
            [
                1.0-q[..., 2, 2]-q[..., 3, 3],
                q[..., 1, 2]-q[..., 3, 0],
                q[..., 1, 3]+q[..., 2, 0]
            ], [
                q[..., 1, 2]+q[..., 3, 0],
                1.0-q[..., 1, 1]-q[..., 3, 3],
                q[..., 2, 3]-q[..., 1, 0]
            ], [
                q[..., 1, 3]-q[..., 2, 0],
                q[..., 2, 3]+q[..., 1, 0],
                1.0-q[..., 1, 1]-q[..., 2, 2]
            ],
        ]
        return _stack_recursive(q, axis=-1)

    def if_true():
        return tf.tile(
            tf.expand_dims(identity_matrix(3, dtype=q0.dtype), 0),
            tf.stack([n.shape[0], 1, 1]))

    return tf.where(cond, if_true(), if_false())


def quaternion_matrix(q):
    """Return homogeneous rotation matrix from quaternion."""
    return homogeneous_rotation_matrix(quaternion_matrix_nh(q))


def quaternion_multiply(q1, q0, axis=-1):
    """Return multiplication of two quaternions."""
    w0, x0, y0, z0 = tf.unstack(q0, axis=axis)
    w1, x1, y1, z1 = tf.unstack(q1, axis=axis)
    return tf.stack([
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0
    ], axis=-1)


def quaternion_conjugate(q):
    """Return the conjugate of quaternion."""
    return tf.concat([q[..., :1], -q[..., 1:]], axis=-1)


def quaternion_inverse(q):
    """Return the inverse of the quaternion."""
    return quaternion_conjugate(q) / tf.expand_dims(vector_norm2(q), axis=-1)


def quaternion_real(q):
    """Return the real part of the quaternion."""
    return q[..., 0]


def quaternion_imag(q):
    """Return the imaginary part of the quaternion."""
    return q[..., 1:]


def vector_norm2(v, axis=-1):
    """Get the vector norm squared along the given axis."""
    return tf.reduce_sum(v**2, axis=axis)


def vector_norm(v, axis=-1):
    """Get the vector norm along the given axis."""
    return tf.sqrt(vector_norm2(v, axis=axis))


def unit_vector(v, axis=-1):
    """Get the unit vector parallel to v, along the specified axis."""
    return v / tf.expand_dims(vector_norm(v, axis=axis), axis=axis)


def outer(a, b):
    """
    Get the outer product of a and b.

    a: shape N + (m,)
    b: shape N + (m,)

    return shape n * m * m, r[..., j, k] = a[..., j] * b[..., k]
    """
    return tf.expand_dims(a, axis=-1) * tf.expand_dims(b, axis=-2)


def vector_product(v0, v1):
    """See documentation for `tf.cross`."""
    return tf.cross(v0, v1)


def angle_between_vectors(v0, v1, directed=True, axis=-1):
    """
    Get the angle between vectors.

    Only directed == True implemented.
    """
    if not directed:
        raise NotImplementedError()

    dot = tf.reduce_sum(v0*v1, axis=-1)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    return tf.acos(dot)


def inverse_matrix(matrix):
    """See documentation to `tf.matrix_inverse`."""
    return tf.matrix_inverse(matrix)


def concatenate_matrices(*matrices):
    """Return the matrix product of each matrix."""
    M = matrices[0]
    for m in matrices[1:]:
        M = tf.matmul(M, m)
    return M


def is_same_transform(matrix0, matrix1):
    """Return a tf.True if the two matrices are the same transform."""
    m0 = matrix0 / matrix0[..., 3:4, 3:4]
    m1 = matrix1 / matrix1[..., 3:4, 3:4]
    return all_close(m0, m1, axis=(-2, -1))


def is_same_quaternion(q0, q1, axis=-1):
    """Return tf.True if two quaternions are equal."""
    return tf.logical_or(
        all_close(q0, q1, axis=axis), all_close(q0, -q1, axis=-1))


# class Transform3D(object):
#     """
#     Class representing a rotation and a shift in 3D space.
#
#     Implemented using tensorflow. Operations should work with batches.
#     """
#
#     @property
#     def rotation_matrix(self):
#         """Matrix representation of this transform."""
#         raise NotImplementedError()
#
#     @property
#     def shift(self):
#         """Vector representation os the associated shift."""
#         raise NotImplementedError()
#
#     def apply_rotation(self, points):
#         """Apply the associated rotation to the points."""
#         return tf.tensordot(self.rotation_matrix, points, axis=[[-1], [-2]])
#
#     def apply_shift(self, points):
#         """Apply the associated shift to the points."""
#         return points + self.shift
#
#     def _apply_to_points(self, points):
#         """Apply the transformation to the given point."""
#         return self.apply_shift(self.apply_rotation(points))
#
#     def _apply_to_transform(self, transform):
#         """Apply the transformation to the other transformation."""
#         if isinstance(transform, _IdentityTransform3D):
#             return self
#         else:
#             if not isinstance(transform, Transform3D):
#                 raise TypeError('transform must be a Transform3D')
#             R = self.apply_rotation(transform.rotation_matrix)
#             t = self.apply_shift(self.apply_rotation(transform.shift))
#             return BaseTransform3D(R, t)
#
#     def apply(self, points_or_transform):
#         """Apply the transformation to the given points_or_transform."""
#         if isinstance(points_or_transform, Transform3D):
#             return self._apply_to_transform(points_or_transform)
#         elif isinstance(points_or_transform, tf.Tensor):
#             return self._apply_to_points(points_or_transform)
#         else:
#             raise TypeError(
#                 'points_or_transform must be a Transform3D or tf.Tensor')
#
#
# class BaseTransform3D(Transform3D):
#     """Base implementation of Transform3D."""
#
#     def __init__(self, rotation_matrix, shift):
#         """Initialize with a Rotation matrix R and displacement vector t."""
#         self._rotation_matrix = rotation_matrix
#         self._shift = shift
#
#     @property
#     def rotation_matrix(self):
#         """Matrix representation of this transform."""
#         return self._rotation_matrix
#
#     @property
#     def shift(self):
#         """Vector representation os the associated shift."""
#         return self._shift
#
#
# class _IdentityTransform3D(Transform3D):
#     """Efficient implementation of Transform3D for identity transform."""
#
#     @property
#     def rotation_matrix(self):
#         raise Exception('Should never need to call this.')
#
#     @property
#     def shift(self):
#         raise Exception('Should never need to call this.')
#
#     def _apply_to_points(self, points):
#         return points
#
#     def _apply_to_transform(self, transform):
#         return transform
