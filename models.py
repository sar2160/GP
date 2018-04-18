import gpflow


def Poisson_Model(X, y, use_priors = False, e_s = 0):

    with gpflow.defer_build():

            like = gpflow.likelihoods.Poisson(binsize = e_s)

            kern_s_effect = gpflow.kernels.RBF(2, active_dims=[1,2], name='space_effect')
            kern_t_effect = gpflow.kernels.RBF(1, active_dims=[0], name='time_effect')
            ## Will have to write custom kernel to match Flaxman 2014
            kern_p_effect = gpflow.kernels.Periodic(1, active_dims=[0], name = 'periodic_effect')
            kern_st_effect = gpflow.kernels.Product([kern_s_effect ,kern_t_effect])

            full_kern =  kern_t_effect + kern_s_effect + kern_p_effect + kern_st_effect


            m = gpflow.models.VGP(X, y, full_kern,  likelihood = like, mean_function = None)

            m.kern.periodic.period = 12
            m.kern.periodic.period.trainable = True

            normal_prior = gpflow.priors.Gaussian(mu = 0 , var = 1)

            if use_priors:
                m.kern.rbf_1.variance.prior    = normal_prior
                m.kern.periodic.variance.prior = normal_prior
                m.kern.rbf_2.variance.prior    = normal_prior

                m.kern.rbf_1.lengthscales.prior = normal_prior
                m.kern.rbf_2.lengthscales.prior = normal_prior
                m.kern.periodic.lengthscales.prior = normal_prior

            return m
