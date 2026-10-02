import torch
import numpy as np

from torch import nn
from vectorization import unvec, unvech, elimination_matrix, duplication_matrix, commutation_matrix, bkron
from torch.nn import functional as F


class ExplicitEncoder(nn.Module):
    def __init__(self,
                 in_dim=3,
                 out_dim=1024,
                 n_primitives=32,
                 cloud_range=torch.tensor([[-1, 1], [-1, 1], [-1, 1]]),
                 init_mode='random',
                 normalized=True,
                 beta=2,
                 diagonal=False,
                 isotropic=False,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.n_basis = out_dim
        self.n_primitives = n_primitives
        self.in_dim = in_dim
        if init_mode == 'uniform':
            dim_coords = []
            for i in range(in_dim):
                coords = torch.linspace(cloud_range[i, 0], cloud_range[i, 1], int(n_primitives**(1/3))),
                dim_coords.append(coords)
            mu = torch.cartesian_prod(*dim_coords)[:n_primitives]
            radii = torch.ones((n_primitives))
        elif init_mode == 'layered':
            n_layers = int(np.log(n_primitives)/np.log(2))
            points = []
            total = 0
            n_layer = 0
            biases = []
            while total < n_primitives:
                n_g = 8**n_layer
                # we don't want the endpoints, hence add 2 and remove them
                x = torch.linspace(cloud_range[0, 0], cloud_range[0, 1], int(2**(n_layer)) + 2)[1:-1]
                y = torch.linspace(cloud_range[1, 0], cloud_range[1, 1], int(2**(n_layer)) + 2)[1:-1]
                z = torch.linspace(cloud_range[2, 0], cloud_range[2, 1], int(2**(n_layer)) + 2)[1:-1]
                cur_points = torch.cartesian_prod(x, y, z)
                points.append(cur_points)
                biases.append(torch.ones(cur_points.shape[0]) / 2**n_layer)
                n_layer += 1
                total += len(cur_points)
                #int(n_g**(1/3))
            mu = torch.cat(points, dim=0)[:n_primitives]
            radii = torch.cat(biases, dim=0)[:n_primitives]
        else: #init_mode = random
            mu = torch.rand((n_primitives, self.in_dim)) * (cloud_range[None, :, 1] - cloud_range[None, :, 0]) + cloud_range[None, :, 0]
            radii = torch.ones((n_primitives))
        self.mu = nn.Parameter(mu, requires_grad=True)
        l_triangs = torch.ones((n_primitives, in_dim*(in_dim+1)//2), dtype=torch.float32) * 0.01
        diagonal_indices = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(np.arange(in_dim, 1, -1, dtype=np.int32))), axis=0)
        l_triangs[:, diagonal_indices] = radii[..., None]
        self.l_triangs = nn.Parameter(l_triangs, requires_grad=True)
        self.weighting = nn.Linear(n_primitives, out_dim)
        self.normalized = normalized
        self.beta = beta
        self.diagonal = diagonal
        self.isotropic = isotropic


    def compute_inv_cov(self):
        l = unvech(self.l_triangs.unsqueeze(-1), self.in_dim)
        inv_cov = l @ l.transpose(-1, -2)
        if self.diagonal:
            mask = torch.eye(self.in_dim, device=inv_cov.device).unsqueeze(0)
            inv_cov = inv_cov * mask
        elif self.isotropic:
            inv_cov = torch.eye(self.in_dim, device=inv_cov.device).unsqueeze(0)
            inv_cov *= self.l_triangs[:, 0]**2

        return inv_cov


    def compute_mahalanobis_distance(self, x, inv_cov):
        x_expanded = x.unsqueeze(2)  # Shape: (batch_size, num_points, 1, 3)
        mu_expanded = self.mu.unsqueeze(0).unsqueeze(
            0
        )  # Shape: (1, 1, num_gaussians, 3)
        precision_expanded = inv_cov.unsqueeze(0).unsqueeze(
            0
        )  # Shape: (1, 1, num_gaussians, 3, 3)

        # breakpoint()

        diff = (
            x_expanded - mu_expanded
        )  # Shape: (batch_size, num_points, num_gaussians, 3)

        diff_transformed = torch.matmul(
            diff.unsqueeze(-2), precision_expanded
        )  # Shape: (batch_size, num_points, num_gaussians, 1, 3)
        diff_transformed = diff_transformed.squeeze(
            -2
        )  # Shape: (batch_size, num_points, num_gaussians, 3)

        mahalanobis_dist = torch.sum(
            diff * diff_transformed, dim=-1
        )  # Shape: (batch_size, num_points, num_gaussians)

        return mahalanobis_dist


    def kernel_fn(self, x, inv_cov):
        '''
        Compute the kernel. Default is a noop.
        '''
        return x


    def forward(self, x):
        '''
        x: (B, N, 3)
        '''
        inv_cov = self.compute_inv_cov()
        mahalanobis_dist = self.compute_mahalanobis_distance(x, inv_cov) # (B, N, G)
        activations = self.kernel_fn(mahalanobis_dist, inv_cov) # (B, N, G)
        projected = self.weighting(activations) # (B, N, K)
        global_feat = torch.amax(projected, dim=-2) # (B, K)
        return global_feat


    def register_mahalanobis_hook(self):
        @torch.no_grad()
        def mu_hook(grad):
            # grad: (G, 3)
            T = unvech(self.l_triangs.unsqueeze(-1).detach(), self.in_dim)
            cov = torch.linalg.inv(T @ T.transpose(-1, -2))
            return (cov @ grad[..., None])[..., 0]

        self.mu.register_hook(mu_hook)


class GaussianEncoder(ExplicitEncoder):
    def kernel_fn(self, x, inv_cov):
        activation = torch.exp(-x / 2)
        if self.normalized:
            activation = activation / torch.sqrt((2 * torch.pi)**self.in_dim * torch.linalg.det(inv_cov))
        return activation


    def register_FIM_hook(self):
        @torch.no_grad()
        def mu_hook(grad):
            # grad: (G, 3)
            T = unvech(self.l_triangs.unsqueeze(-1).detach(), self.in_dim)
            cov = torch.linalg.inv(T @ T.transpose(-1, -2))
            return (cov @ grad[..., None])[..., 0]

        @torch.no_grad()
        def cov_hook(grad):
            # grad: (G, 6)
            device = self.l_triangs.device
            T = unvech(self.l_triangs.unsqueeze(-1).detach(), self.in_dim)
            K = commutation_matrix(self.in_dim).to(device)[None]
            D = duplication_matrix(self.in_dim).to(device)[None]
            L = elimination_matrix(self.in_dim).to(device)[None]
            I_d = torch.eye(self.in_dim, device=device)[None]
            N = (K + torch.eye(self.in_dim**2, device=device)) / 2
            left = L @ bkron(I_d, T) @ L.transpose(-1, -2)
            right = left.transpose(-1, -2)
            middle = torch.linalg.inv(L @ N @ L.transpose(-1, -2))
            preconditioner = left @ middle @ right / 2

            return (preconditioner @ grad[..., None])[..., 0]

        self.mu.register_hook(mu_hook)
        self.l_triangs.register_hook(cov_hook)


def jacobian_tester(mu, l_triangs, weighting_weight, weighting_bias, sample, in_dim):
    '''
    Consider the INPUTS to be fixed. Then compute the jacobian of the output w.r.t the model params
    For now, let's assume we have a single output channel of the weighting. This should make everything more manageable
    '''
    def compute_inv_cov():
        l = unvech(l_triangs.unsqueeze(-1), in_dim)
        return l @ l.transpose(-1, -2)


    def compute_mahalanobis_distance(x, inv_cov):
        x_expanded = x.unsqueeze(2)  # Shape: (batch_size, num_points, 1, 3)
        mu_expanded = mu.unsqueeze(0).unsqueeze(
            0
        )  # Shape: (1, 1, num_gaussians, 3)
        precision_expanded = inv_cov.unsqueeze(0).unsqueeze(
            0
        )  # Shape: (1, 1, num_gaussians, 3, 3)

        # breakpoint()

        diff = (
            x_expanded - mu_expanded
        )  # Shape: (batch_size, num_points, num_gaussians, 3)

        diff_transformed = torch.matmul(
            diff.unsqueeze(-2), precision_expanded
        )  # Shape: (batch_size, num_points, num_gaussians, 1, 3)
        diff_transformed = diff_transformed.squeeze(
            -2
        )  # Shape: (batch_size, num_points, num_gaussians, 3)

        mahalanobis_dist = torch.sum(
            diff * diff_transformed, dim=-1
        )  # Shape: (batch_size, num_points, num_gaussians)

        return mahalanobis_dist


    def kernel_fn(x, inv_cov):
        '''
        Compute the kernel. Default is a noop.
        '''
        return torch.exp(-x / 2)

    inv_cov = compute_inv_cov()
    mahalanobis_dist = compute_mahalanobis_distance(sample, inv_cov) # (B, N, G)
    activations = kernel_fn(mahalanobis_dist, inv_cov) # (B, N, G)
    projected = F.linear(activations, weighting_weight, weighting_bias)
    return torch.mean(projected, dim=(0, 1))


if __name__ == '__main__':
    model = GaussianEncoder()
    # model.register_FIM_hook()
    model.register_mahalanobis_hook()
    cloud = torch.ones((128, 1024, 3), dtype=torch.float32)
    model(cloud)
    print(model.compute_inv_cov()[0])
