from models_utility.function_gp import lt_log_determinant
from torch import triangular_solve
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions import kl_divergence
from torch.nn import functional as F
import gpytorch

torch.set_default_tensor_type(torch.DoubleTensor)

zitter = 1e-8


class RFF_GPLVM(nn.Module):
    def __init__(self, num_batch, num_sample_pt, param_dict, Y, device=None, ifPCA=True):
        super(RFF_GPLVM, self).__init__()
        self.device = device
        self.name = None
        self.num_batch = num_batch
        self.num_samplept = num_sample_pt  # L/2
        self.latent_dim = param_dict['latent_dim']  # Q
        self.N = param_dict['N']                    # !!!
        self.num_m = param_dict['num_m']            # m
        self.noise = param_dict['noise_err']
        self.lr_hyp = param_dict['lr_hyp']
        self.Q = self.latent_dim

        self.Y = Y
        
        self.total_num_sample = self.num_samplept * self.num_m  # m * L/2
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.L_div_2 = self.total_num_sample
        self.K = self.num_m
        self.L = torch.tensor(self.L_div_2*2)

        if self.num_m==1:
            # if SE kernel is used, then self.mu = 0, and requires_grad=False
            self.mu = nn.Parameter(torch.zeros(self.num_m, self.latent_dim, device=self.device), requires_grad=False)  # shape: K * Q
        else:
            self.mu = nn.Parameter(torch.zeros(self.num_m, self.latent_dim, device=self.device), requires_grad=True)  # shape: K * Q

        self.log_std = nn.Parameter(torch.ones(self.num_m, self.latent_dim, device=self.device), requires_grad=True)  # shape: k * Q
        
        if ifPCA:
            pca = PCA(n_components=self.latent_dim)
            X = pca.fit_transform(self.Y)
        else:
            X = torch.randn(self.N, self.latent_dim, device=self.device)
            
        if ifPCA and X_init is None:
            pca = PCA(n_components=self.latent_dim)
            X = pca.fit_transform(self.Y)
        elif X_init is not None: 
            X = X_init  
        else: 
            X = torch.randn(self.N, self.latent_dim, device=self.device)

        self.mu_x = nn.Parameter(torch.tensor(X, device=self.device), requires_grad=True)    # shape: N * Q
        self.log_sigma_x = nn.Parameter(torch.randn(self.N, self.latent_dim, device=self.device), requires_grad=True)
        
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
        

        for i_th in range(self.L_div_2):  # TODO: check if it can be improved without using for
            SM_eps = torch.randn(1, self.latent_dim, device=self.device)
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

                Phistar_i_th =  self.num_samplept**0.5 * torch.cat([xstar_spectral.cos(), xstar_spectral.sin()], 1)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)  #  N * (m * L),  N_star * (M * L)


    def _compute_gram_approximate(self, Phi):  # shape:  (m*L) x (m*L)
        return Phi.t() @ Phi + (self.likelihood.noise + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()


    def _compute_gram_approximate_2(self, Phi):  # shape:  N x N
        return Phi @ Phi.T


    def _kl_div_qp(self):

        # shape: N x Q
        q_dist = torch.distributions.Normal(loc=self.mu_x, scale= F.softplus(self.log_sigma_x))
        p_dist = torch.distributions.Normal(loc=torch.zeros_like(q_dist.loc), scale=torch.ones_like(q_dist.loc))

        return kl_divergence(q_dist, p_dist).sum().div(self.N * self.latent_dim)

    def compute_loss(self, batch_y, kl_option):
        """
        :param batch_y:
        :return: approximate lower bound of negative log marginal likelihood
        """
        obs_dim = batch_y.shape[1]
        obs_num = batch_y.shape[0]
        batch_y = torch.tensor(batch_y, device=self.device, dtype=torch.double)
        Phi = self._compute_sm_basis()

        # negative log-marginal likelihood
        if Phi.shape[0]>Phi.shape[1]:  # if N > (m*L)
            Approximate_gram = self._compute_gram_approximate(Phi)  # shape:  (m * L） x  (m * L）
            L = torch.cholesky(Approximate_gram)
            Lt_inv_Phi_y = triangular_solve((Phi.t()).matmul(batch_y), L, upper=False)[0]

            # todo: need to double-check this part
            neg_log_likelihood = (0.5 / self.likelihood.noise) * (batch_y.pow(2).sum() - Lt_inv_Phi_y.pow(2).sum())
            neg_log_likelihood += lt_log_determinant(L)
            neg_log_likelihood += (-self.total_num_sample) * 2 * self.likelihood.noise.sqrt()
            neg_log_likelihood += 0.5 * obs_num * (np.log(2 * np.pi) + 2 * self.likelihood.noise.sqrt())

        else:
            k_matrix = self._compute_gram_approximate_2(Phi=Phi) # shape: N x N
            C_matrix = k_matrix + self.likelihood.noise * torch.eye(self.N, device=self.device)
            L = torch.cholesky(C_matrix) # shape: N x N
            L_inv_y = triangular_solve(batch_y, L, upper=False)[0]

            # compute log-likelihood by ourselves
            constant_term = 0.5 * obs_num * np.log(2 * np.pi) * obs_dim
            log_det_term = torch.diagonal(L, dim1=-2, dim2=-1).sum().log() * obs_dim
            yy = 0.5 * L_inv_y.pow(2).sum()
            neg_log_likelihood = (constant_term + log_det_term + yy).div(obs_dim * obs_num)

        if kl_option:
            kl_x = self._kl_div_qp().div(self.N * 50)
            loss_all = neg_log_likelihood + kl_x
        else:
            loss_all = neg_log_likelihood

        return loss_all
    
    # -----------------------------------------------------------------------------
    # Inference `Z`, `V`, `alpha`.
    # -----------------------------------------------------------------------------
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