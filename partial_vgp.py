import tensorflow as tf

from gpflow.models.vgp import VGP
from gpflow.decors import autoflow, params_as_tensors
from gpflow import settings
from gpflow.conditionals import conditional




class PartialVGP(VGP):
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, K_part=None):

        if K_part is None:
            K_part = self.kern
        K = K_part
    
            
        mu, var = conditional(Xnew, self.X, K, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, white=True)
        return mu + self.mean_function(Xnew), var
    
    @autoflow((settings.float_type, [None, None]))
    def predict_f_linear(self, Xnew, full_cov=False):
        kern = self.kern.linear
        return self._build_predict(Xnew, full_cov, kern)

    @autoflow((settings.float_type, [None, None]))
    def predict_f_periodic(self, Xnew, full_cov=False):
        kern = self.kern.periodic
        return self._build_predict(Xnew, full_cov, kern)

    @autoflow((settings.float_type, [None, None]))
    def predict_f_rbf(self, Xnew, full_cov=False):
        kern = self.kern.rbf
        return self._build_predict(Xnew, full_cov, kern)