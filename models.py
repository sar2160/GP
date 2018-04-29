import gpflow
from partial_vgp import PartialVGP


def Poisson_Model(X, y, use_priors = False, e_s = 0, period = 12):

    with gpflow.defer_build():

            like = gpflow.likelihoods.Poisson(binsize = e_s)

            kern_s_effect = gpflow.kernels.RBF(2, active_dims=[1,2], name='space_effect')
            kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')
            ## Will have to write custom kernel to match Flaxman 2014
            kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
            kern_st_effect = gpflow.kernels.Product([kern_s_effect ,kern_t_effect])

            full_kern =  kern_t_effect + kern_s_effect + kern_p_effect + kern_st_effect


            m = gpflow.models.VGP(X, y, full_kern,  likelihood = like, mean_function = None)

            m.kern.periodic.period = period
            m.kern.periodic.period.trainable = True

            normal_prior = gpflow.priors.Gaussian(mu = 0 , var = 1)

            if use_priors:
                m.kern.rbf_1.variance.prior    = normal_prior
                m.kern.periodic.variance.prior = normal_prior
                m.kern.rbf_2.variance.prior    = normal_prior

                #m.kern.rbf_1.lengthscales.prior = normal_prior
                #m.kern.rbf_2.lengthscales.prior = normal_prior
                #m.kern.periodic.lengthscales.prior = normal_prior

            return m

def Matern12_Model(X, y, use_priors = False, e_s = 0, period = 12 ):

    with gpflow.defer_build():

        like = gpflow.likelihoods.Poisson(binsize = e_s)
        kern_s_effect = gpflow.kernels.Matern12(input_dim =  2, active_dims=[1,2], name='space_effect')
        kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')

        ## Will have to write custom kernel to match Flaxman 2014
        kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
        kern_st_effect = gpflow.kernels.Product([kern_s_effect ,kern_t_effect])

        full_kern =  kern_t_effect + kern_s_effect + kern_p_effect + kern_st_effect

        m = gpflow.models.VGP(X, y, full_kern,  likelihood = like, mean_function = None)

        m.kern.periodic.period = period
        m.kern.periodic.period.trainable = True


        t_prior = gpflow.priors.StudentT(mean = 0 , scale = 1, deg_free = 4)

        if use_priors:
                m.kern.matern12.variance.prior    = t_prior
                m.kern.periodic.variance.prior = t_prior
                m.kern.rbf.variance.prior    = t_prior

                #m.kern.matern12.lengthscales.prior = t_prior
                #m.kern.rbf.lengthscales.prior = t_prior
                #m.kern.periodic.lengthscales.prior = t_prior

    return m

def SafeMatern32_Model(X, y, use_priors = False, e_s = 0, period = 12 , partial=False):

    with gpflow.defer_build():

        like = gpflow.likelihoods.Poisson(binsize = e_s)
        kern_s_effect = gpflow.kernels.SafeMatern32(input_dim =  2, active_dims=[1,2], name='space_effect')
        kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')

        ## Will have to write custom kernel to match Flaxman 2014
        kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
        kern_st_effect = gpflow.kernels.Product([kern_s_effect ,kern_t_effect])

        full_kern =  kern_t_effect + kern_s_effect + kern_p_effect + kern_st_effect
        
        if partial:
            m = PartialVGP(X, y, full_kern,  likelihood = like, mean_function = None)
        else:
            m = gpflow.models.VGP(X, y, full_kern,  likelihood = like, mean_function = None)

        m.kern.periodic.period = period
        m.kern.periodic.period.trainable = True


        t_prior = gpflow.priors.StudentT(mean = 0 , scale = 5, deg_free = 4)

        if use_priors:
                m.kern.safematern32.variance.prior    = t_prior
                m.kern.periodic.variance.prior = t_prior
                m.kern.rbf.variance.prior    = t_prior

                m.kern.safematern32.lengthscales.prior = t_prior
                m.kern.rbf.lengthscales.prior = t_prior
                m.kern.periodic.lengthscales.prior = t_prior

    return m


def All_Matern_Model(X, y, use_priors = False, e_s = 0, period = 12 ):

    with gpflow.defer_build():

        like = gpflow.likelihoods.Poisson(binsize = e_s)
        kern_s_effect = gpflow.kernels.SafeMatern32(input_dim =  2, active_dims=[1,2], name='space_effect')
        kern_t_effect = gpflow.kernels.Matern52(1, active_dims=[0], name='time_effect')

        ## Will have to write custom kernel to match Flaxman 2014
        kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
        kern_st_effect = gpflow.kernels.Product([kern_s_effect ,kern_t_effect])

        full_kern =  kern_t_effect + kern_s_effect + kern_p_effect + kern_st_effect

        m = gpflow.models.VGP(X, y, full_kern,  likelihood = like, mean_function = None)

        m.kern.periodic.period = period
        m.kern.periodic.period.trainable = True


        t_prior = gpflow.priors.StudentT(mean = 0 , scale = 1, deg_free = 4)

        if use_priors:
                m.kern.safematern32.variance.prior    = t_prior
                m.kern.periodic.variance.prior = t_prior
                m.kern.matern52.variance.prior    = t_prior

                #m.kern.safematern32.lengthscales.prior = t_prior
                #m.kern.matern52.lengthscales.prior = t_prior
                #m.kern.periodic.lengthscales.prior = t_prior

    return m

def LongTerm_Model(X, y, use_priors = False, e_s = 0, period = 12):

    with gpflow.defer_build():

            like = gpflow.likelihoods.Poisson(binsize = e_s)

            kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')
            ## Will have to write custom kernel to match Flaxman 2014
            kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
            kern_l_effect = gpflow.kernels.Linear(1, active_dims = [0], name = 'linear_effect', variance = 1.0)

            full_kern =  kern_t_effect + kern_p_effect + kern_l_effect


            m = gpflow.models.VGP(X, y, full_kern,  likelihood = like, mean_function = None)

            m.kern.periodic.period = period
            m.kern.periodic.period.trainable = True

            Tprior = gpflow.priors.StudentT(mean = 0 , scale = 1, deg_free = 4)

            if use_priors:
                m.kern.rbf.variance.prior      = Tprior
                m.kern.periodic.variance.prior = Tprior
                m.kern.linear.variance.prior   = Tprior
                
                m.kern.rbf.lengthscales.prior = Tprior
                m.kern.periodic.lengthscales.prior = Tprior
                
            return m


def LongTermPartial_Model(X, y, use_priors = False, e_s = 0, period = 12):

    with gpflow.defer_build():

            like = gpflow.likelihoods.Poisson(binsize = e_s)

            kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')
            ## Will have to write custom kernel to match Flaxman 2014
            kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
            kern_l_effect = gpflow.kernels.Linear(1, active_dims = [0], name = 'linear_effect', variance = 1.0)

            full_kern =  kern_t_effect + kern_p_effect + kern_l_effect


            m = PartialVGP(X, y, full_kern,  likelihood = like, mean_function = None)

            m.kern.periodic.period = period
            m.kern.periodic.period.trainable = True

            Tprior = gpflow.priors.StudentT(mean = 0 , scale = 1, deg_free = 4)

            if use_priors:
                m.kern.rbf.variance.prior      = Tprior
                m.kern.periodic.variance.prior = Tprior
                m.kern.linear.variance.prior   = Tprior
                
                m.kern.rbf.lengthscales.prior = Tprior
                m.kern.periodic.lengthscales.prior = Tprior
                
            return m

        
def Poisson_Model_T(X, y, use_priors = False, e_s = 0, period = 12, partial=False):

    with gpflow.defer_build():

            like = gpflow.likelihoods.Poisson(binsize = e_s)

            kern_s_effect = gpflow.kernels.RBF(2, active_dims=[1,2], name='space_effect')
            kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')
            ## Will have to write custom kernel to match Flaxman 2014
            kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
            kern_st_effect = gpflow.kernels.Product([kern_s_effect ,kern_t_effect])

            full_kern =  kern_t_effect + kern_s_effect + kern_p_effect + kern_st_effect

            if partial:
                m = PartialVGP(X, y, full_kern,  likelihood = like, mean_function = None)
            else:
                m = gpflow.models.vgp(X, y, full_kern,  likelihood = like, mean_function = None)


            m.kern.periodic.period = period
            m.kern.periodic.period.trainable = False

            t_prior = gpflow.priors.StudentT(mean = 0 , scale = 1, deg_free = 4)

            if use_priors:
                m.kern.rbf_1.variance.prior    = t_prior
                m.kern.periodic.variance.prior = t_prior
                m.kern.rbf_2.variance.prior    = t_prior

                #m.kern.rbf_1.lengthscales.prior = t_prior
                #m.kern.rbf_2.lengthscales.prior = t_prior
                #m.kern.periodic.lengthscales.prior = t_prior

            return m
        
        
def ST_Model(X, y, use_priors = False, e_s = 0):

    with gpflow.defer_build():

            like = gpflow.likelihoods.Poisson(binsize = e_s)

            kern_s_effect = gpflow.kernels.RBF(2, active_dims=[1,2], name='space_effect')
            kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')

            full_kern =  kern_t_effect + kern_s_effect

            m = gpflow.models.VGP(X, y, full_kern,  likelihood = like, mean_function = None)

 

            t_prior = gpflow.priors.StudentT(mean = 0 , scale = 1, deg_free = 4)

            if use_priors:
                m.kern.rbf_1.variance.prior    = t_prior
                m.kern.rbf_2.variance.prior    = t_prior

                m.kern.rbf_1.lengthscales.prior = t_prior
                m.kern.rbf_2.lengthscales.prior = t_prior

            return m
        
        
def HMC_LongTerm_Model(X, y, use_priors = False, e_s = 0, period = 12):

    with gpflow.defer_build():

            like = gpflow.likelihoods.Poisson(binsize = e_s)

            kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')
            ## Will have to write custom kernel to match Flaxman 2014
            kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
            kern_l_effect = gpflow.kernels.Linear(1, active_dims = [0], name = 'linear_effect', variance = 1.0)

            full_kern =  kern_t_effect + kern_p_effect + kern_l_effect


            m = gpflow.models.GPMC(X, y, full_kern,  likelihood = like, mean_function = None, num_latent =0)

            m.kern.periodic.period = period
            m.kern.periodic.period.trainable = True

            Tprior = gpflow.priors.StudentT(mean = 0 , scale = 1, deg_free = 4)
            gamma  = gpflow.priors.Gamma(1., 1.)
            if use_priors:
                m.kern.rbf.variance.prior      = Tprior
                m.kern.periodic.variance.prior = Tprior
                m.kern.linear.variance.prior   = Tprior
                
                #m.kern.rbf.lengthscales.prior      = gamma
                #m.kern.periodic.lengthscales.prior = gamma
                
            return m