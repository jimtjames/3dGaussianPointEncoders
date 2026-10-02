import numpy as np
import torch
import torch.nn as nn
from vectorization import unvec, unvech, elimination_matrix, duplication_matrix, commutation_matrix, bkron
from torch.nn import functional as F


class GaussianEncoder(nn.Module):
    def __init__(
        self,
        in_dim=3,
        out_dim=1024,
        n_primitives=32,
        cloud_range=torch.tensor([[-1, 1], [-1, 1], [-1, 1]]),
        init_mode="random",
        normalized=True,
        beta=2,
        diagonal=False,
        isotropic=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if in_dim != 3:
            raise ValueError("This implementation assumes in_dim == 3.")

        self.n_basis = out_dim
        self.n_primitives = n_primitives
        self.in_dim = in_dim

        self.normalized = normalized
        self.beta = beta

        self.diagonal = diagonal
        self.isotropic = isotropic

        cloud_range = torch.as_tensor(
            cloud_range,
            dtype=torch.float32,
        )
        self.register_buffer(
            "cloud_range",
            cloud_range.clone(),
        )

        if init_mode == "uniform":
            dim_coords = []

            for i in range(in_dim):
                coords = torch.linspace(
                    self.cloud_range[i, 0],
                    self.cloud_range[i, 1],
                    int(n_primitives ** (1 / 3)),
                )
                dim_coords.append(coords)

            mu = torch.cartesian_prod(*dim_coords)[:n_primitives]
            radii = torch.ones(mu.shape[0])

            if mu.shape[0] != n_primitives:
                raise ValueError(
                    f"uniform initialization produced {mu.shape[0]} "
                    f"points for n_primitives={n_primitives}. "
                    "n_primitives should normally be a perfect cube "
                    "for init_mode='uniform'."
                )

        elif init_mode == "layered":
            points = []
            biases = []

            total = 0
            n_layer = 0

            while total < n_primitives:
                x = torch.linspace(
                    self.cloud_range[0, 0],
                    self.cloud_range[0, 1],
                    2 ** n_layer + 2,
                )[1:-1]

                y = torch.linspace(
                    self.cloud_range[1, 0],
                    self.cloud_range[1, 1],
                    2 ** n_layer + 2,
                )[1:-1]

                z = torch.linspace(
                    self.cloud_range[2, 0],
                    self.cloud_range[2, 1],
                    2 ** n_layer + 2,
                )[1:-1]

                cur_points = torch.cartesian_prod(x, y, z)

                points.append(cur_points)
                biases.append(
                    torch.ones(cur_points.shape[0]) / 2 ** n_layer
                )

                total += cur_points.shape[0]
                n_layer += 1

            mu = torch.cat(points, dim=0)[:n_primitives]
            radii = torch.cat(biases, dim=0)[:n_primitives]

        else:
            mu = (
                torch.rand((n_primitives, self.in_dim))
                * (
                    self.cloud_range[None, :, 1]
                    - self.cloud_range[None, :, 0]
                )
                + self.cloud_range[None, :, 0]
            )

            radii = torch.ones(n_primitives)

        self.mu = nn.Parameter(mu, requires_grad=True)

        l_triangs = torch.ones(
            (
                n_primitives,
                in_dim * (in_dim + 1) // 2,
            ),
            dtype=torch.float32,
        ) * 0.01

        diagonal_indices = np.concatenate(
            (
                np.zeros(1, dtype=np.int32),
                np.cumsum(
                    np.arange(
                        in_dim,
                        1,
                        -1,
                        dtype=np.int32,
                    )
                ),
            ),
            axis=0,
        )

        l_triangs[:, diagonal_indices] = radii[..., None]

        self.l_triangs = nn.Parameter(
            l_triangs,
            requires_grad=True,
        )

        self.weighting = nn.Linear(
            n_primitives,
            out_dim,
        )

        self.likelihood = self.gaussian_likelihood

        self.register_buffer(
            "inv_cov",
            None,
            persistent=False,
        )

        self.register_buffer(
            "cov",
            None,
            persistent=False,
        )

        self.register_buffer(
            "eigs",
            None,
            persistent=False,
        )

        self.K = None
        self.max_gaussians_per_voxel = None

        self.register_buffer(
            "voxel_size",
            None,
            persistent=False,
        )

        self.register_buffer(
            "voxel_to_gaussians",
            None,
            persistent=False,
        )

        self.register_buffer(
            "padded_voxel_to_gaussians",
            None,
            persistent=False,
        )

        self.register_buffer(
            "padding_mask",
            None,
            persistent=False,
        )

        self.register_buffer(
            "batch_indices_base",
            None,
            persistent=False,
        )

        self.register_buffer(
            "point_indices_base",
            None,
            persistent=False,
        )

        self.register_buffer(
            "bboxes",
            None,
            persistent=False,
        )

        self.register_buffer(
            "factor",
            None,
            persistent=False,
        )

    def compute_inv_cov(self):
        """
        Construct the Gaussian precision matrices.

        Returns:
            inv_cov: (G, 3, 3)
        """
        l = unvech(
            self.l_triangs.unsqueeze(-1),
            self.in_dim,
        )

        inv_cov = l @ l.transpose(-1, -2)

        if self.diagonal:
            mask = torch.eye(
                self.in_dim,
                device=inv_cov.device,
                dtype=inv_cov.dtype,
            ).unsqueeze(0)

            inv_cov = inv_cov * mask

        elif self.isotropic:
            identity = torch.eye(
                self.in_dim,
                device=inv_cov.device,
                dtype=inv_cov.dtype,
            ).unsqueeze(0)

            # Written this way instead of using *= because the latter
            # cannot safely broadcast from (1, 3, 3) -> (G, 3, 3).
            inv_cov = (
                identity
                * self.l_triangs[:, 0, None, None] ** 2
            )

        return inv_cov

    def gaussian_likelihood(self, xyz):
        """
        Compute the Gaussian activation for every point/Gaussian pair.

        Args:
            xyz: (B, N, 3)

        Returns:
            likelihoods: (B, N, G)
        """
        B, N, C = xyz.shape

        xyz_exp = xyz.unsqueeze(2)  # (B, N, 1, C)

        means_exp = (
            self.mu
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, G, C)

        diffs = xyz_exp - means_exp  # (B, N, G, C)

        # Match the old behavior:
        #
        # training -> recompute precision
        # eval     -> use cached precision
        if self.training or self.inv_cov is None:
            inv_cov = self.compute_inv_cov()
        else:
            inv_cov = self.inv_cov

        # (B, N, G, C) @ (G, C, C)
        quad_form = torch.einsum(
            "bngc,gcd->bngd",
            diffs,
            inv_cov,
        )

        # d^T P d
        quad_form = torch.einsum(
            "bngc,bngc->bng",
            quad_form,
            diffs,
        )

        return torch.exp(-0.5 * quad_form)

    def gaussian_likelihood_dist_prune(self, xyz):
        """
        Compute Gaussian activations only for point/Gaussian pairs
        satisfying the precomputed Euclidean distance heuristic.
        """
        B, N, C = xyz.shape
        G = self.n_primitives

        factor = (
            self.factor
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, G)

        xyz_exp = xyz.unsqueeze(2)  # (B, N, 1, C)

        means_exp = (
            self.mu
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, G, C)

        diffs = xyz_exp - means_exp  # (B, N, G, C)

        # Squared Euclidean distance.
        dists = torch.sum(
            diffs ** 2,
            dim=-1,
        )  # (B, N, G)

        conditional = dists < factor

        (
            batch_indices,
            point_indices,
            gaussian_indices,
        ) = torch.nonzero(
            conditional,
            as_tuple=True,
        )

        inv_cov_selected = self.inv_cov[
            gaussian_indices
        ]  # (T, C, C)

        # Reuse the difference tensor computed for the distance test.
        diff_valid = diffs[
            batch_indices,
            point_indices,
            gaussian_indices,
        ]  # (T, C)

        diff_transform = torch.matmul(
            diff_valid.unsqueeze(1),
            inv_cov_selected,
        ).squeeze(1)

        mahalanobis_term_valid = (
            diff_valid * diff_transform
        ).sum(dim=-1)

        likelihoods = xyz.new_zeros(
            (B, N, G)
        )

        likelihoods[
            batch_indices,
            point_indices,
            gaussian_indices,
        ] = torch.exp(
            -0.5 * mahalanobis_term_valid
        )

        return likelihoods

    def gaussian_likelihood_bbox_prune(self, xyz):
        """
        Compute Gaussian activations only for point/Gaussian pairs
        whose point falls inside the Gaussian's axis-aligned bbox.
        """
        B, N, C = xyz.shape
        G = self.n_primitives

        xyz_expanded = xyz.unsqueeze(2)
        # (B, N, 1, C)

        lower_bounds = self.bboxes[..., :, 0]
        # (G, C)

        upper_bounds = self.bboxes[..., :, 1]
        # (G, C)

        condition = (
            (lower_bounds <= xyz_expanded)
            & (xyz_expanded <= upper_bounds)
        )  # (B, N, G, C)

        all_dim_inside = torch.all(
            condition,
            dim=3,
        )  # (B, N, G)

        (
            batch_indices,
            point_indices,
            gaussian_indices,
        ) = torch.nonzero(
            all_dim_inside,
            as_tuple=True,
        )

        X_expanded_valid = xyz[
            batch_indices,
            point_indices,
        ]

        mu_selected_valid = self.mu[
            gaussian_indices
        ]

        inv_cov_selected_valid = self.inv_cov[
            gaussian_indices
        ]

        diff_valid = (
            X_expanded_valid
            - mu_selected_valid
        )

        diff_transform = torch.matmul(
            diff_valid.unsqueeze(1),
            inv_cov_selected_valid,
        ).squeeze(1)

        mahalanobis_term_valid = (
            diff_valid * diff_transform
        ).sum(dim=-1)

        likelihoods = xyz.new_zeros(
            (B, N, G)
        )

        likelihoods[
            batch_indices,
            point_indices,
            gaussian_indices,
        ] = torch.exp(
            -0.5 * mahalanobis_term_valid
        )

        return likelihoods

    def gaussian_likelihood_voxel_prune_vectorized(
        self,
        xyz,
    ):
        """
        Compute Gaussian activations using the precomputed padded
        voxel -> Gaussian lookup.

        Important:
            B and N must match the B/N passed to voxel_prune().
            This intentionally retains the original implementation's
            fixed-shape cached indexing optimization.
        """
        B, N, C = xyz.shape
        G = self.n_primitives

        # --------------------------------------------------------------
        # 1. Compute voxel index for every point
        # --------------------------------------------------------------

        voxel_indices = (
            (xyz - self.cloud_range[:, 0])
            / self.voxel_size
        ).long()

        voxel_indices = voxel_indices.clamp(
            0,
            self.K - 1,
        )

        voxel_flat_index = (
            voxel_indices[..., 0] * self.K ** 2
            + voxel_indices[..., 1] * self.K
            + voxel_indices[..., 2]
        )

        # --------------------------------------------------------------
        # 2. Retrieve padded Gaussian lists
        # --------------------------------------------------------------

        padded_gaussian_indices = (
            self.padded_voxel_to_gaussians[
                voxel_flat_index
            ]
        )
        # (B, N, Max_G)

        valid_combinations_mask = (
            self.padding_mask[
                voxel_flat_index
            ]
        )
        # (B, N, Max_G)

        # --------------------------------------------------------------
        # 3. Convert padded representation to flat valid combinations
        # --------------------------------------------------------------

        gaussian_indices_selected = (
            padded_gaussian_indices
        )

        batch_indices_valid = (
            self.batch_indices_base[
                valid_combinations_mask
            ]
        )

        point_indices_valid = (
            self.point_indices_base[
                valid_combinations_mask
            ]
        )

        gaussian_indices_valid = (
            gaussian_indices_selected[
                valid_combinations_mask
            ]
        )

        # --------------------------------------------------------------
        # 4. Sparse Mahalanobis calculation
        # --------------------------------------------------------------

        X_expanded_valid = xyz[
            batch_indices_valid,
            point_indices_valid,
        ]

        mu_selected_valid = self.mu[
            gaussian_indices_valid
        ]

        inv_cov_selected_valid = self.inv_cov[
            gaussian_indices_valid
        ]

        diff_valid = (
            X_expanded_valid
            - mu_selected_valid
        )

        diff_transform = torch.matmul(
            diff_valid.unsqueeze(1),
            inv_cov_selected_valid,
        ).squeeze(1)

        mahalanobis_term_valid = (
            diff_valid * diff_transform
        ).sum(dim=-1)

        # --------------------------------------------------------------
        # 5. Scatter sparse values back into (B, N, G)
        # --------------------------------------------------------------

        likelihoods = xyz.new_zeros(
            (B, N, G)
        )

        likelihoods[
            batch_indices_valid,
            point_indices_valid,
            gaussian_indices_valid,
        ] = torch.exp(
            -0.5 * mahalanobis_term_valid
        )

        return likelihoods

    @torch.no_grad()
    def _prepare_padded_voxel_to_gaussians(self):
        """
        Convert binary voxel_to_gaussians:
            (K^3, G)

        into:
            padded_voxel_to_gaussians: (K^3, Max_G)
            padding_mask:              (K^3, Max_G)
        """
        K3 = self.K ** 3

        max_gaussians_per_voxel = 0
        gaussian_indices_list = []

        for voxel_index in range(K3):
            indices = torch.nonzero(
                self.voxel_to_gaussians[
                    voxel_index
                ]
            ).flatten()

            gaussian_indices_list.append(
                indices
            )

            max_gaussians_per_voxel = max(
                max_gaussians_per_voxel,
                len(indices),
            )

        padded_gaussians = torch.full(
            (
                K3,
                max_gaussians_per_voxel,
            ),
            -1,
            dtype=torch.long,
            device=self.mu.device,
        )

        padding_masks = torch.zeros(
            (
                K3,
                max_gaussians_per_voxel,
            ),
            dtype=torch.bool,
            device=self.mu.device,
        )

        for voxel_index in range(K3):
            indices = gaussian_indices_list[
                voxel_index
            ]

            num_indices = len(indices)

            if num_indices > 0:
                padded_gaussians[
                    voxel_index,
                    :num_indices,
                ] = indices

                padding_masks[
                    voxel_index,
                    :num_indices,
                ] = True

        self.padded_voxel_to_gaussians = (
            padded_gaussians
        )

        self.padding_mask = padding_masks

        self.max_gaussians_per_voxel = (
            max_gaussians_per_voxel
        )

    @torch.no_grad()
    def voxel_prune(
        self,
        K,
        threshold,
        n_rand=50000,
        B=1,
        N=2048,
    ):
        """
        Build the voxel -> Gaussian lookup.

        B and N describe the inference shape whose batch/point index
        tensors should be cached.
        """
        self._ensure_eval_cache()

        device = self.mu.device
        dtype = self.mu.dtype

        # This deliberately mirrors the slightly unusual original
        # sampling expression exactly:
        #
        #   rand = U * (min - max) + max
        #
        # which is still uniform over [min, max].
        ranges = (
            self.cloud_range[:, 0]
            - self.cloud_range[:, 1]
        )

        rand = (
            torch.rand(
                (n_rand, 3),
                device=device,
                dtype=dtype,
            )
            * ranges
            + self.cloud_range[:, 1]
        )

        self.voxel_size = (
            self.cloud_range[:, 1]
            - self.cloud_range[:, 0]
        ) / K

        # --------------------------------------------------------------
        # Voxel index of each random point
        # --------------------------------------------------------------

        voxel_indices = (
            (rand - self.cloud_range[:, 0])
            / self.voxel_size
        ).long()

        voxel_indices = voxel_indices.clamp(
            0,
            K - 1,
        )

        voxel_flat_index = (
            voxel_indices[..., 0] * K ** 2
            + voxel_indices[..., 1] * K
            + voxel_indices[..., 2]
        )

        # --------------------------------------------------------------
        # Weighted Gaussian activation at random points
        # --------------------------------------------------------------

        likelihoods = self.compute_density(
            rand.unsqueeze(0)
        ).squeeze(0)

        gaussian_maxes = torch.amax(
            torch.abs(
                self.weighting.weight.data
            ),
            dim=0,
        )

        likelihoods *= gaussian_maxes

        # --------------------------------------------------------------
        # Maximum weighted activation observed in each voxel
        # --------------------------------------------------------------

        voxel_maxes = torch.zeros(
            (
                K ** 3,
                self.n_primitives,
            ),
            device=device,
            dtype=likelihoods.dtype,
        )

        for i in range(n_rand):
            voxel_maxes[
                voxel_flat_index[i],
                :,
            ] = torch.maximum(
                likelihoods[i, :],
                voxel_maxes[
                    voxel_flat_index[i],
                    :,
                ],
            )

        self.K = K

        self.voxel_to_gaussians = (
            voxel_maxes >= threshold
        ).to(torch.uint8)

        self.likelihood = (
            self.gaussian_likelihood_voxel_prune_vectorized
        )

        self._prepare_padded_voxel_to_gaussians()

        # --------------------------------------------------------------
        # Original fixed-B/N cached indexing optimization
        # --------------------------------------------------------------

        M = self.max_gaussians_per_voxel

        self.batch_indices_base = (
            torch.arange(
                B,
                device=device,
            )
            .view(B, 1, 1)
            .repeat(1, N, M)
        )

        self.point_indices_base = (
            torch.arange(
                N,
                device=device,
            )
            .view(1, N, 1)
            .repeat(B, 1, M)
        )

        return voxel_maxes

    def compute_density(self, x):
        """
        Equivalent to the old compute_density().

        Assumes self.inv_cov has already been cached.
        """
        x_expanded = x.unsqueeze(2)

        mu_expanded = (
            self.mu
            .unsqueeze(0)
            .unsqueeze(0)
        )

        precision_expanded = (
            self.inv_cov
            .unsqueeze(0)
            .unsqueeze(0)
        )

        diff = (
            x_expanded
            - mu_expanded
        )

        diff_transformed = torch.matmul(
            diff.unsqueeze(-2),
            precision_expanded,
        )

        diff_transformed = (
            diff_transformed.squeeze(-2)
        )

        mahalanobis_dist = torch.sum(
            diff * diff_transformed,
            dim=-1,
        )

        return torch.exp(
            -0.5 * mahalanobis_dist
        )

    @torch.no_grad()
    def get_bboxes(
        self,
        threshold,
        cov,
    ):
        """
        Original bbox construction heuristic.
        """
        bboxes = torch.zeros(
            (
                self.n_primitives,
                3,
                2,
            ),
            device=self.mu.device,
            dtype=self.mu.dtype,
        )

        max_alpha = torch.amax(
            torch.abs(
                self.weighting.weight.data
            ),
            dim=0,
        )

        factor = (
            -2
            * torch.log(
                threshold / max_alpha
            )
        ).unsqueeze(-1)

        diff = torch.sqrt(
            factor
            * torch.diagonal(
                cov,
                dim1=-2,
                dim2=-1,
            )
        )

        bboxes[..., 0] = self.mu - diff
        bboxes[..., 1] = self.mu + diff

        self.bboxes = bboxes

        return bboxes

    @torch.no_grad()
    def bbox_prune(
        self,
        threshold,
    ):
        """
        Prepare bbox pruning and switch forward() to the bbox path.
        """
        self._ensure_eval_cache()

        bboxes = self.get_bboxes(
            threshold,
            self.cov,
        )

        self.likelihood = (
            self.gaussian_likelihood_bbox_prune
        )

        return bboxes

    @torch.no_grad()
    def set_dist_threshold(
        self,
        threshold,
    ):
        """
        Original Euclidean-distance threshold heuristic.
        """
        max_alpha = torch.amax(
            torch.abs(
                self.weighting.weight.data
            ),
            dim=0,
        )

        factor = (
            -2
            * torch.log(
                threshold / max_alpha
            )
            * self.eigs
        )

        self.factor = factor

        return factor

    @torch.no_grad()
    def dist_prune(
        self,
        threshold,
    ):
        """
        Prepare distance pruning and switch forward() to the distance path.
        """
        self._ensure_eval_cache()

        factor = self.set_dist_threshold(
            threshold
        )

        self.likelihood = (
            self.gaussian_likelihood_dist_prune
        )

        return factor

    def clear_pruning(self):
        self.likelihood = (
            self.gaussian_likelihood
        )

    @torch.no_grad()
    def _refresh_eval_cache(self):
        """
        Cache Gaussian-only quantities exactly once for evaluation.
        """
        self.inv_cov = self.compute_inv_cov()

        # New parameterization stores precision directly, so covariance
        # has to be reconstructed here.
        self.cov = torch.linalg.inv(
            self.inv_cov
        )

        # Largest covariance eigenvalue.
        self.eigs = torch.linalg.eigvalsh(
            self.cov
        )[:, -1]

    @torch.no_grad()
    def _ensure_eval_cache(self):
        if (
            self.inv_cov is None
            or self.cov is None
            or self.eigs is None
        ):
            self._refresh_eval_cache()

    def eval(self):
        # Put all child modules into eval mode first.
        super().eval()

        # Then cache Gaussian geometry.
        self._refresh_eval_cache()

        return self

    def train(self, mode=True):
        if mode:
            # Match the old behavior: training always restores dense
            # Gaussian evaluation.
            self.likelihood = (
                self.gaussian_likelihood
            )

            # Invalidate derived inference quantities.
            self.inv_cov = None
            self.cov = None
            self.eigs = None

        return super().train(mode)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3)

        Returns:
            global_feat: (B, out_dim)
        """
        activations = self.likelihood(x)
        # (B, N, G)

        projected = self.weighting(
            activations
        )
        # (B, N, K)

        global_feat = torch.amax(
            projected,
            dim=-2,
        )
        # (B, K)

        return global_feat

    def register_mahalanobis_hook(self):
        @torch.no_grad()
        def mu_hook(grad):
            T = unvech(
                self.l_triangs
                .unsqueeze(-1)
                .detach(),
                self.in_dim,
            )

            cov = torch.linalg.inv(
                T @ T.transpose(-1, -2)
            )

            return (
                cov @ grad[..., None]
            )[..., 0]

        self.mu.register_hook(
            mu_hook
        )

    def register_FIM_hook(self):
        @torch.no_grad()
        def mu_hook(grad):
            T = unvech(
                self.l_triangs
                .unsqueeze(-1)
                .detach(),
                self.in_dim,
            )

            cov = torch.linalg.inv(
                T @ T.transpose(-1, -2)
            )

            return (
                cov @ grad[..., None]
            )[..., 0]

        @torch.no_grad()
        def cov_hook(grad):
            device = self.l_triangs.device

            T = unvech(
                self.l_triangs
                .unsqueeze(-1)
                .detach(),
                self.in_dim,
            )

            K = commutation_matrix(
                self.in_dim
            ).to(device)[None]

            D = duplication_matrix(
                self.in_dim
            ).to(device)[None]

            L = elimination_matrix(
                self.in_dim
            ).to(device)[None]

            I_d = torch.eye(
                self.in_dim,
                device=device,
            )[None]

            N = (
                K
                + torch.eye(
                    self.in_dim ** 2,
                    device=device,
                )
            ) / 2

            left = (
                L
                @ bkron(I_d, T)
                @ L.transpose(-1, -2)
            )

            right = left.transpose(
                -1,
                -2,
            )

            middle = torch.linalg.inv(
                L
                @ N
                @ L.transpose(-1, -2)
            )

            preconditioner = (
                left
                @ middle
                @ right
                / 2
            )

            return (
                preconditioner
                @ grad[..., None]
            )[..., 0]

        self.mu.register_hook(
            mu_hook
        )

        self.l_triangs.register_hook(
            cov_hook
        )
