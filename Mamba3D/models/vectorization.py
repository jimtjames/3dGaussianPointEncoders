import torch
import numpy as np


def commutation_matrix(n: int) -> torch.Tensor:
    '''
    Constructs the commutation matrix K_{n,n}
    s.t. K_{n.n} vec(A) = vec(A^T)
    where A in mbb{R}^{n times n}
    '''
    # source: wikipedia
    # determine permutation applied by K
    w = np.arange(n * n).reshape((n, n), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return torch.eye(n * n)[w, :]


def duplication_matrix(n: int) -> torch.Tensor:
    """
    Constructs the duplication matrix D_n ∈ ℝ^{n^2 × (n(n+1)/2)}.
    Maps vech(S) to vec(S) for symmetric S, using column-major vech order.
    """
    D = torch.zeros((n * n, (n * (n + 1)) // 2), dtype=torch.float32)

    idx = 0
    for j in range(n):       # column
        for i in range(j, n):  # row ≥ col (lower triangle)
            flat_ij = i * n + j  # position in vec(S)
            flat_ji = j * n + i  # symmetric position
            if i == j:
                D[flat_ij, idx] = 1.0
            else:
                D[flat_ij, idx] = 1.0
                D[flat_ji, idx] = 1.0
            idx += 1
    return D


def elimination_matrix(n: int) -> torch.Tensor:
    """
    Constructs the elimination matrix L_n ∈ ℝ^{(n(n+1)/2) × n^2},
    such that vech(A) = L_n @ vec(A), for column-major vec and vech.
    """
    tril_rows, tril_cols = torch.tril_indices(n, n)
    flat_indices = tril_cols * n + tril_rows  # column-major linear indices

    # Sort to get correct vech column-major order
    sorted_idx = flat_indices.argsort()
    vec_indices = flat_indices[sorted_idx]
    k = len(vec_indices)

    L = torch.zeros((k, n * n), dtype=torch.float32)
    L[torch.arange(k), vec_indices] = 1.0
    return L


def vec(X: torch.Tensor) -> torch.Tensor:
    """
    Vectorize a matrix (or batch of matrices) by stacking **columns**, i.e., column-major order.
    Input: X ∈ ℝ^{n × m} or ℝ^{... × n × m}
    Output: vec(X) ∈ ℝ^{n*m} or ℝ^{... × n*m}
    """
    return X.transpose(-2, -1).reshape(*X.shape[:-2], -1, 1)


def vech(X: torch.Tensor) -> torch.Tensor:
    """
    Half-vectorize a symmetric matrix (or batch) by extracting the lower triangular part including diagonal.
    Input: X ∈ ℝ^{n × n} or ℝ^{... × n × n}
    Output: vech(X) ∈ ℝ^{n(n+1)/2} or ℝ^{... × n(n+1)/2}
    """
    n = X.shape[-1]
    rows, cols = torch.tril_indices(n, n, device=X.device)
    # Get sort indices for column-major ordering
    order = (cols * n + rows).argsort()
    return X[..., rows[order], cols[order], None]


def unvec(v: torch.Tensor, n: int) -> torch.Tensor:
    """
    Reshape column-stacked vector v ∈ ℝ^{n^2} into n×n matrix.
    Assumes vec uses column-major ordering.
    """
    return v.reshape(-1, n, n).transpose(-1, -2)


# def unvech(vh: torch.Tensor, n: int) -> torch.Tensor:
    # """
    # Reconstruct symmetric n×n matrix from vech-style vector vh ∈ ℝ^{n(n+1)/2}, column-major order.
    # """
    # mat = torch.zeros((n, n), dtype=vh.dtype, device=vh.device)
    # tril_rows, tril_cols = torch.tril_indices(n, n)

    # flat_indices = tril_cols * n + tril_rows
    # sorted_idx = flat_indices.argsort()

    # mat[tril_rows[sorted_idx], tril_cols[sorted_idx]] = vh
    # mat[tril_cols[sorted_idx], tril_rows[sorted_idx]] = vh  # symmetric
    # return mat

# def unvech(vh: torch.Tensor, n: int) -> torch.Tensor:
    # """
    # Reconstruct symmetric n×n matrices from vech-style vectors (column-major),
    # supporting batched input: vh shape (..., n(n+1)/2).
    # Returns: symmetric matrix of shape (..., n, n)
    # """
    # tril_rows, tril_cols = torch.tril_indices(n, n, device=vh.device)
    # flat_idx = tril_cols * n + tril_rows
    # sorted = flat_idx.argsort()
    # i = tril_rows[sorted]
    # j = tril_cols[sorted]

    # linear1 = i * n + j
    # linear2 = j * n + i
    # all_indices = torch.cat([linear1, linear2], dim=0)  # (2k,)

    # vh_double = torch.cat([vh, vh], dim=-1)  # (..., 2k)

    # # Flatten batch dims
    # batch_shape = vh.shape[:-1]
    # print(batch_shape)
    # B = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
    # vh_flat = vh_double.view(B, -1)

    # # Build output buffer
    # out = torch.zeros((B, n * n), dtype=vh.dtype, device=vh.device)
    # out.scatter_add_(1, all_indices.unsqueeze(0).expand(B, -1), vh_flat)

    # mat = out.view(*batch_shape, n, n)

    # # Correct for doubled diagonal (added twice)
    # diag_mask = torch.eye(n, dtype=torch.bool, device=vh.device)
    # mat[..., diag_mask] /= 2

    # return mat


def unvech(vh: torch.Tensor, n: int) -> torch.Tensor:
    """
    Reconstruct symmetric n×n matrix from vech-style vector vh ∈ ℝ^{n(n+1)/2}, column-major order.
    """
    mat = torch.zeros((vh.shape[0], n, n), dtype=vh.dtype, device=vh.device)
    tril_rows, tril_cols = torch.tril_indices(n, n, device=vh.device)

    flat_indices = tril_cols * n + tril_rows
    sorted_idx = flat_indices.argsort()

    mat[:, tril_rows[sorted_idx], tril_cols[sorted_idx]] = vh[..., 0]
    mat[:, tril_cols[sorted_idx], tril_rows[sorted_idx]] = vh[..., 0]  # symmetric
    return mat
# def unvech(vh: torch.Tensor, n: int) -> torch.Tensor:
    # """
    # Reconstruct symmetric n×n matrix from vech-style vector vh ∈ ℝ^{n(n+1)/2},
    # preserving autograd by avoiding in-place operations.
    # """

    # D = duplication_matrix(n)[None].to(vh.device).to(vh.dtype)
    # vec_sym = D @ vh
    # return unvec(vec_sym, n)

# def unvech(vh: torch.Tensor, n: int) -> torch.Tensor:
    # """
    # Reconstruct symmetric n×n matrix from its vech (lower-triangular part).
    # Assumes vech uses column-major ordering.
    # """
    # mat = torch.zeros((vh.shape[0], n, n), dtype=vh.dtype, device=vh.device)
    # tril_rows, tril_cols = torch.tril_indices(n, n)

    # # Compute flat indices in column-major order and sort
    # flat_indices = tril_cols * n + tril_rows
    # sorted_idx = flat_indices.argsort()

    # # Fill lower triangle
    # mat[..., tril_rows[sorted_idx], tril_cols[sorted_idx]] = vh[..., 0]

    # # Reflect to upper triangle
    # mat[..., tril_cols[sorted_idx], tril_rows[sorted_idx]] = vh[..., 0]

    # return mat


def bkron(x, y):
    '''
    x: (B, a, b)
    y: (B, c, d)
    Batched kronecker of the last two dimensions hence:
    Out: (B, a*c, b*d)
    '''
    return torch.einsum('iab,icd->iacbd', x, y).reshape((-1, x.shape[-2] * y.shape[-2], x.shape[-1] * y.shape[-1]))


if __name__ == '__main__':
    d = 3
    D = duplication_matrix(d)[None]
    a = torch.rand((d, d))
    b = (a + a.T)[None].repeat(4, 1, 1)
    print(torch.allclose(D @ vech(b), vec(b)))
    L = elimination_matrix(d)[None]
    print(torch.allclose(L @ vec(b), vech(b)))
    print(torch.allclose(unvec(vec(b), d), b))
    print(torch.allclose(unvech(vech(b), d), b))
    K = commutation_matrix(d)[None]
    print(torch.allclose(K @ vec(b), vec(b.transpose(-1, -2))))


