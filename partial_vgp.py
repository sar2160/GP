import tensorflow as tf

from gpflow.models.vgp import VGP
from gpflow.decors import autoflow, params_as_tensors, name_scope
from gpflow import settings
from gpflow.conditionals import conditional




class PartialVGP(VGP):
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, K_part=None):

        if K_part is None:
            K_part = self.kern
    
        mu, var = my_conditional(Xnew, self.X, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, white=True, K_part = K_part)
        return mu + self.mean_function(Xnew), var
    
    @autoflow((settings.float_type, [None, None]))
    def predict_f_linear(self, Xnew, full_cov=False):
        kern = self.kern.linear
        return self._build_predict(Xnew, full_cov=full_cov, K_part=kern)

    @autoflow((settings.float_type, [None, None]))
    def predict_f_periodic(self, Xnew, full_cov=False):
        kern = self.kern.periodic
        return self._build_predict(Xnew, full_cov=full_cov, K_part =kern)

    @autoflow((settings.float_type, [None, None]))
    def predict_f_rbf(self, Xnew, full_cov=False):
        kern = self.kern.rbf
        return self._build_predict(Xnew, full_cov=full_cov, K_part=kern)
    
    
    
@name_scope()
def my_conditional(Xnew, X, kern, f, *, full_cov=False, q_sqrt=None, white=False,K_part=None):
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N(0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case `f` represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).

    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

    :param Xnew: data matrix, size N x D.
    :param X: data points, size M x D.
    :param kern: GPflow kernel.
    :param f: data matrix, M x K, representing the function values at X,
        for K functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x K or K x M x M.
    :param white: boolean of whether to use the whitened representation as
        described above.

    :return: two element tuple with conditional mean and variance.
    """
    num_data = tf.shape(X)[0]  # M
    Kmm = kern.K(X) + tf.eye(num_data, dtype=settings.float_type) * settings.numerics.jitter_level
    if K_part is not None:
        print('Using kernel component {}'.format(K_part.name))
        Kmn = K_part.K(X, Xnew) # changed
    else:
        Kmn = kern.K(X, Xnew)
    
    if full_cov:
        Knn = kern.K(Xnew)
    else:
        Knn = kern.Kdiag(Xnew)
    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)


@name_scope()
def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False):
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # K
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)
        shape = tf.stack([num_func, 1, 1])
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # K x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # K x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar
