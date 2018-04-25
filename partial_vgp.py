import tensorflow as tf

from gpflow.vgp import VGP
from gpflow.param import AutoFlow
from gpflow._settings import settings

float_type = settings.dtypes.float_type


class PartialVGP(VGP):
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, K_part=None):

        if K_part is None:
            K_part = self.kern
        K = K_part

        mu, var = conditional(Xnew, self.X, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, white=True)
        return mu + self.mean_function(Xnew), var


    @AutoFlow(Xnew=(float_type, [None, None]))
    def predict_f_partial(self, Xnew, full_cov=False, kern=None):
        return self.build_predict(Xnew, full_cov, kern)
