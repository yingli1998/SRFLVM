"""============================================================================
Dataset loading functions.
============================================================================"""

"""============================================================================
Dataset loading functions.
============================================================================"""

from .dataset import Dataset
from GPy import kern
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import (expit as logistic,
                           logsumexp)
from sklearn.datasets import (make_blobs,
                              make_circles,
                              make_moons,
                              make_s_curve,
                              load_breast_cancer)

# -----------------------------------------------------------------------------

# if 'gwg3' in os.getcwd():
#     BASE_DIR = f'{REMOTE_DIR}/datasets'
# else:
#     BASE_DIR = f'{LOCAL_DIR}/datasets'

BASE_DIR = '/root/gplvm-sm-proj/'

# -----------------------------------------------------------------------------

def load_dataset(rng, name, emissions, test_split=0):
    """Given a dataset string, returns data and possibly true generative
    parameters.
    """
    loader = {
        'bridges': load_bridges,
        '3PhData': load_3PhData,
        'cifar': load_cifar,
        'cmu': load_cmu,
        'cmu1': load_cmu1,
        'cmu2': load_cmu2,
        'cmu3': load_cmu3,
        'cmu4': load_cmu4,
        'congress': load_congress,
        'mnist_big': load_mnist_big,
        'covid': load_covid,
        'fiji': load_fiji,
        'highschool': load_highschool,
        'hippo': load_hippocampus,
        'lorenz': load_lorenz,
        'mnist': load_mnist,
        'mnistb': load_mnistb,
        'montreal': load_montreal,
        'newsgroups': load_20newsgroups,
        'simdata1': load_simdata,
        'simdata2': load_simdata,
        'simdata3': load_simdata,
        'spam': load_spam,
        's-curve-torch': gen_s_curve_torch,
        's-curve': gen_s_curve,
        's-curve-batch': gen_s_curve_batch,
        'spikes': load_spikes,
        'yale': load_yale,
        'ovarian': load_ovarian,
        'cancer': load_cancer,
        'exchange': load_exchange, 
        'bface': load_bface,
    }[name]

    if name == 's-curve' or name == 's-curve-batch' or name == 's-curve-torch':
        return loader(rng, emissions, test_split)
    else:
        return loader(rng, test_split)


# -----------------------------------------------------------------------------
def gen_s_curve(rng, emissions, test_split):
    """Generate synthetic data from datasets generating process.
    """
    N = 500
    J = 100
    D = 2
    
    print(N, J)

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]

    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------

    K = kern.RBF(input_dim=D, lengthscale=1).K(X)
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using `F` and/or `K`.
    # ----------------------------------------
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)
    if emissions == 'gaussian':
        Y = F + np.random.normal(5, scale=0.5, size=F.shape)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)
    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t,
                       test_split=test_split)
    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J + 1, dtype=float)
        Y = rng.negative_binomial(R, 1 - P)
        return Dataset(rng, 's-curve', False, False, Y=Y, X=X, F=F, R=R,
                       latent_dim=D, labels=t, test_split=test_split)
    else:
        assert (emissions == 'poisson')
        print("Poission")
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)


# -----------------------------------------------------------------------------
def gen_s_curve_batch(rng, emissions, test_split):
    """Generate synthetic data from datasets generating process.
    """
    batch_size = 77
    N = 500
    J = 100
    D = 2

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]

    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------

    K = kern.RBF(input_dim=D, lengthscale=1).K(X)

    Yb = np.empty((batch_size, N, J))
    Fb = np.empty((batch_size, N, J))
    if emissions == 'negbinom':
        Rb = np.empty((batch_size, J))

    for i in range(batch_size):
        F = rng.multivariate_normal(np.zeros(N), K, size=J).T

        # Generate emissions using `F` and/or `K`.
        # ----------------------------------------
        if emissions == 'bernoulli':
            P = logistic(F)
            Y = rng.binomial(1, P).astype(np.double)
        if emissions == 'gaussian':
            Y = F + np.random.normal(0, scale=0.5, size=F.shape)
        elif emissions == 'multinomial':
            C = 100
            pi = np.exp(F - logsumexp(F, axis=1)[:, None])
            Y = np.zeros(pi.shape)
            for n in range(N):
                Y[n] = rng.multinomial(C, pi[n])
        elif emissions == 'negbinom':
            P = logistic(F)
            R = np.arange(1, J + 1, dtype=float)
            Y = rng.negative_binomial(R, 1 - P)
        else:
            assert (emissions == 'poisson')
            theta = np.exp(F)
            Y = rng.poisson(theta)

        Yb[i] = Y
        Fb[i] = F
        if emissions == 'negbinom':
            Rb[i] = R
    if emissions == 'bernoulli':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)
    if emissions == 'gaussian':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)
    elif emissions == 'multinomial':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)
    elif emissions == 'negbinom':
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, Rb, t,
                       test_split=test_split)
    else:
        assert (emissions == 'poisson')
        return Dataset(rng, 's-curve', False, Yb, X, Fb, K, None, t,
                       test_split=test_split)


# -----------------------------------------------------------------------------
def gen_s_curve_torch(rng, emissions, test_split):

    from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
    import torch

    """Generate synthetic data from datasets generating process.
    """
    N = 500
    J = 100
    D = 2

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]


    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------

    RBF_cov = ScaleKernel(RBFKernel())
    Period_cov = ScaleKernel(PeriodicKernel())
    # parameter setting
    RBF_cov.outputscale = 0.5
    RBF_cov.base_kernel.lengthscale = 1
    #
    Period_cov.outputscale = 0.5
    Period_cov.base_kernel.lengthscale = 1.0
    Period_cov.base_kernel.period_length = 4.5  # setting
    # Period_cov.base_kernel.period_length = 5    # setting 1
    # Period_cov.base_kernel.period_length = 4    # setting 2

    # K = Period_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy()

    # K = RBF_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy()

    K = RBF_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy() + \
        Period_cov(torch.tensor(X)).add_jitter(1e-5).evaluate().detach().numpy()

    # K = kern.RBF(input_dim=D, lengthscale=1).K(X)



    '''# -------------------------------------------------------'''
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using `F` and/or `K`.
    # ----------------------------------------
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

    if emissions == 'gaussian':
        Y = F + np.random.normal(5, scale=0.5, size=F.shape)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J + 1, dtype=float)
        Y = rng.negative_binomial(R, 1 - P)
        return Dataset(rng, 's-curve', False, False, Y=Y, X=X, F=F, R=R, latent_dim=D, labels=t, test_split=test_split)

    else:
        assert (emissions == 'poisson')
        print("Poission")
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t, test_split=test_split)

