from torch import triangular_solve
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions import kl_divergence
from torch.nn import functional as F
import gpytorch
import sys 
from models.func_gp import lt_log_determinant
from  pypolyagamma import PyPolyaGamma
from torch.distributions.multivariate_normal import MultivariateNormal

from scipy.linalg.lapack import dpotrs
from scipy.linalg import solve_triangular
from torch.special import expit as ag_logistic
from untity import sample_gaussian

torch.set_default_tensor_type(torch.DoubleTensor)

zitter = 1e-8

class RFF_RFLVM_Bernoulli(nn.Module):
    def __init__(self, num_batch, num_sample_pt, param_dict, Y, device=None, ifPCA=True, X_init=None):
        super(RFF_RFLVM_Bernoulli, self).__init__()
        self.device = device
        self.name = None
        self.num_batch = num_batch
        self.num_samplept = num_sample_pt  # L/2
        self.latent_dim = param_dict['latent_dim']  # Q
        self.N = param_dict['N']                    # !!!
        self.M = param_dict['M'] 
        self.num_m = param_dict['num_m']            # m
        self.noise = param_dict['noise_err']
        self.lr_hyp = param_dict['lr_hyp']
        self.L = self.num_m*self.num_samplept*2
        self.Q = self.latent_dim
        
        self.with_a = True

        self.Y = torch.tensor(Y)
        
        self.total_num_sample = self.num_samplept * self.num_m  # m * L/2
        # self.likelihood = Gaussian(variance=self.noise, device=device) if likelihood == None else likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        self.L_div_2 = self.total_num_sample
        self.K = self.num_m
        self.L = torch.tensor(self.L_div_2*2)

        # shape: m * 1
        self.log_weight = nn.Parameter(torch.randn(self.num_m, 1, device=self.device), requires_grad=False)

        if self.num_m==1:
            # if SE kernel is used, then self.mu = 0, and requires_grad=False
            self.mu = nn.Parameter(torch.zeros(self.num_m, self.latent_dim, device=self.device), requires_grad=False)  # shape: m * Q
        else:
            self.mu = nn.Parameter(torch.zeros(self.num_m, self.latent_dim, device=self.device), requires_grad=False)  # shape: m * Q

        self.log_std = nn.Parameter(torch.ones(self.num_m, self.latent_dim, device=self.device), requires_grad=False)  # shape: m * Q

        if ifPCA:
            pca = PCA(n_components=self.latent_dim)
            X = pca.fit_transform(self.Y)
        else:
            X = torch.randn(self.N, self.latent_dim, device=self.device)

        self.mu_x = nn.Parameter(torch.tensor(X, device=self.device), requires_grad=True)    # shape: N * Q
        self.log_sigma_x = nn.Parameter(torch.randn(self.N, self.latent_dim, device=self.device), requires_grad=True)
        
        # hyperparameter for A
        bias_var = 10.
        bias_var = torch.tensor(bias_var)
        prior_Sigma         = torch.eye(self.L)
        prior_Sigma[-1, -1] = torch.sqrt(bias_var)
        self.inv_S          = torch.inverse(prior_Sigma)
        mu_A                = torch.zeros(self.L)
        self.inv_S_mu_A     = self.inv_S @ mu_A
        
        # init A and omega
        self.omega  = torch.zeros(self.Y.shape)
        self.A = torch.normal(0, 1, (self.M, self.L))    
        self.pg = PyPolyaGamma()
        
        #         # Initialize cluster assignments and counts.
        self.Z = np.random.choice(self.K, size=self.L_div_2)   # K * L/2
        self.Z_phi_pre = torch.zeros(self.L_div_2, self.K, device=self.device) 
        for l in range(self.L_div_2):
            self.Z_phi_pre[l, self.Z[l]] = 1
        self.Z_phi = nn.Parameter(self.Z_phi_pre, requires_grad=True)        
        
        # Initialize the parameters of V
        self.va = torch.ones((1, self.K))
        self.vb = torch.ones((1, self.K))
        self.v = torch.ones((1, self.K))

        # Initialize the parameters of alpha
        self.alpha_a0 = 3.
        self.alpha_b0 = 1 / 3.
        self.alpha_a = self.alpha_a0
        self.alpha_b = self.alpha_b0
        self.alpha = 1
        

    def _compute_sm_basis(self, x_star=None, f_eval=False):
        multiple_Phi = []
        current_sampled_spectral_list = []

        if f_eval:  # use to evaluate the latent function 'f'
            x = self.mu_x
        else:
            std = F.softplus(self.log_sigma_x)   # shape: N * Q
            eps = torch.randn_like(std)          # don't preselect/prefix it in __init__ function
            x = self.mu_x + eps * std

        SM_std = F.softplus(self.log_std)
        mu_L = (self.mu.T @ self.Z_phi[:self.L_div_2, :].T).T
        std_L = (SM_std.T @ self.Z_phi[:self.L_div_2, :].T).T

        for i_th in range(self.num_m):  # TODO: check if it can be improved without using for
            SM_eps = torch.randn(self.num_samplept, self.latent_dim, device=self.device)
            sampled_spectral_pt = mu_L[i_th] + std_L[i_th] * SM_eps  # L/2 * Q

            if x_star is not None:
                current_sampled_spectral_list.append(sampled_spectral_pt)

            x_spectral = (2 * np.pi) * x.matmul(sampled_spectral_pt.t())    # N * L/2

            Phi_i_th = (2 / self.L).sqrt() * torch.cat([x_spectral.cos(), x_spectral.sin()], 1)

            multiple_Phi.append(Phi_i_th)

        if x_star is None:
            return torch.cat(multiple_Phi, 1)  #  N * (m * L）

        else:
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list):
                xstar_spectral = (2 * np.pi) * x_star.matmul(current_sampled.t())

                Phistar_i_th = (SM_weight[i_th] / self.num_samplept).sqrt() * torch.cat([xstar_spectral.cos(), xstar_spectral.sin()], 1)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)  #  N * (m * L),  N_star * (M * L)


    def _compute_gram_approximate(self, Phi):  # shape:  (m*L) x (m*L)
        return Phi.t() @ Phi + (self.likelihood.noise + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()


    def _compute_gram_approximate_2(self, Phi):  # shape:  N x N
        return Phi @ Phi.T
    
    def _a_func(self, m=None):
        if m is not None:
            return self.Y[:, m]
        return self.Y

    def _b_func(self, m=None):
        if m is not None:
            return torch.ones(self.Y[:, m].shape)
        return torch.ones(self.Y.shape)
    
    def _kappa_func(self, m):
        return self._a_func(m) - (self._b_func(m) / 2.0)


    def _kl_div_qp(self):

        # shape: N x Q
        q_dist = torch.distributions.Normal(loc=self.mu_x, scale= F.softplus(self.log_sigma_x))
        p_dist = torch.distributions.Normal(loc=torch.zeros_like(q_dist.loc), scale=torch.ones_like(q_dist.loc))

        return kl_divergence(q_dist, p_dist).sum().div(self.N * self.latent_dim)
    
    def sample_omega(self, b, psi, omega): 
        b = b.cpu().detach().numpy()
        psi = psi.cpu().detach().numpy()
        omega = omega.cpu().detach().numpy()
        self.pg.pgdrawv(b.ravel(),
                        psi.ravel(),
                        omega.ravel())
        return omega.reshape(self.Y.shape)
        

    def compute_loss(self, batch_y, kl_option):
        """
        :param batch_y:
        :return: approximate lower bound of negative log marginal likelihood
        """
        obs_dim = batch_y.shape[1]
        obs_num = batch_y.shape[0]
        batch_y = torch.tensor(batch_y, device=self.device, dtype=torch.double)
    
        Phi = self._compute_sm_basis()
        self.K = Phi @ Phi.T
        
        # close form omega
        psi = Phi @ self.A.T
        b  = torch.ones(self.Y.shape)
        self.omega = torch.tensor(self.sample_omega(b, psi, self.omega), requires_grad=False)

        # close form a 
        for m in range(self.M):
            J = (Phi * self.omega[:, m][:, None]).T @ Phi + self.inv_S
            h = Phi.T @ self._kappa_func(m) + self.inv_S_mu_A
            joint_sample = sample_gaussian(J=J, h=h) # test
            self.A[m] = torch.tensor(joint_sample, requires_grad=False)
        
        phi_a = Phi @ self.A.T
        P     = ag_logistic(phi_a)
        LL    = self.Y * torch.log(P) + (1 - self.Y) * torch.log(1 - P)
        neg_log_likelihood = LL.sum()

        
        if kl_option:
            kl = self._kl_div_qp()
            loss_all = neg_log_likelihood + kl
        else:
            loss_all = neg_log_likelihood

        return loss_all
    
    def compute_zphi_loss(self, batch_y):
        E_log_v = torch.special.digamma(self.va) - torch.special.digamma(self.va + self.vb)         # (1, K)
        E_log_1_minus_v = torch.special.digamma(self.vb) - torch.special.digamma(self.va + self.vb) # (1, K)
        
        neg_log_likelihood = self.compute_loss(batch_y, kl_option=False)
        
        total_loss = neg_log_likelihood
                
        for l in range(self.L_div_2):
            for k in range(self.K):
                phi_lk = F.relu(self.Z_phi[l, k].clone()) + 1e-6 
                phi_lk_sq = phi_lk ** 2

                # E_q[log v_k]
                term1 = E_log_v[0, k]

                # \sum_{j=1}^k E_q[log(1 - v_j)]
                term2 = torch.sum(E_log_1_minus_v[0, :k+1]) 
                
                entropy_term = -phi_lk * torch.log(phi_lk+1e-6)  # 避免 log(0)
                
                loss_lk = phi_lk_sq * (term1 + term2) + entropy_term
                
                total_loss += loss_lk

        return total_loss
        
    @torch.no_grad()
    def inference_v(self):
        for k in range(self.K):
            va_k = 1 + sum(self.Z_phi[:, k])
            vb_k = self.alpha + self.Z_phi[:, k + 1:].sum()
            self.va[:, k] = va_k
            self.vb[:, k] = vb_k
            self.v[:, k] = va_k / (va_k + vb_k)

    @torch.no_grad()
    def inference_alpha(self):
        ect_v_k = 0
        for k in range(self.K):
            ect_v_k += torch.special.digamma(self.vb[:, k]) - torch.special.digamma(self.va[:, k] + self.vb[:, k])
        self.alpha_a = self.alpha_a0
        self.alpha_b = self.alpha_b0 - ect_v_k
        self.alpha = self.alpha_a / self.alpha_b