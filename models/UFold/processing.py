"""
RNA Secondary Structure Processing Module

This module provides efficient GPU-accelerated preprocessing and postprocessing
functions for RNA secondary structure prediction using deep learning models.
"""

import torch
import torch.nn.functional as F


class RNAConstants:
    """Constants for RNA processing."""
    BASE_TO_IDX = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    IDX_TO_BASE = ['A', 'U', 'C', 'G']
    PAIRING_STRENGTHS = {'AU': 2, 'UA': 2, 'GC': 3, 'CG': 3, 'UG': 0.8, 'GU': 0.8}


def _create_pairing_matrix(device: torch.device) -> torch.Tensor:
    """Create base pairing strength lookup table."""
    pairing_strengths = torch.zeros(4, 4, device=device)
    for pair, strength in RNAConstants.PAIRING_STRENGTHS.items():
        i, j = RNAConstants.BASE_TO_IDX[pair[0]], RNAConstants.BASE_TO_IDX[pair[1]]
        pairing_strengths[i, j] = strength
    return pairing_strengths


def _compute_weighted_sum(mat: torch.Tensor, i: torch.Tensor, j: torch.Tensor, 
                         t_vals: torch.Tensor, exp_weights: torch.Tensor,
                         offset_i: int, offset_j: int, n: int) -> torch.Tensor:
    """Compute weighted sum for matrix calculations."""
    i_expanded = i.unsqueeze(2)
    j_expanded = j.unsqueeze(2)
    t_expanded = t_vals.unsqueeze(0).unsqueeze(0)
    
    # Boundary checks and clamping
    valid_mask = ((i_expanded + offset_i * t_expanded >= 0) & 
                  (i_expanded + offset_i * t_expanded < n) &
                  (j_expanded + offset_j * t_expanded >= 0) & 
                  (j_expanded + offset_j * t_expanded < n))
    
    i_clamp = torch.clamp(i_expanded + offset_i * t_expanded, 0, n-1)
    j_clamp = torch.clamp(j_expanded + offset_j * t_expanded, 0, n-1)
    
    # Gather values and apply weights
    weighted_vals = torch.where(valid_mask, mat[i_clamp, j_clamp], 0.0)
    weighted_vals *= exp_weights.unsqueeze(0).unsqueeze(0)
    
    # Handle zero masking
    padded_vals = F.pad(weighted_vals, (0, 1))
    first_zero = torch.argmax((padded_vals == 0).float(), dim=2)
    mask = torch.arange(len(t_vals), device=mat.device).unsqueeze(0).unsqueeze(0) >= first_zero.unsqueeze(2)
    
    return torch.where(mask, 0.0, weighted_vals).sum(dim=2)


# ===============================
# PREPROCESSING FUNCTIONS
# ===============================

def creatmat(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Create optimized RNA pairing matrix with GPU acceleration.
    
    Args:
        data: One-hot encoded RNA sequence
        device: Target device for computation
        
    Returns:
        Combined pairing strength matrix (m1 + m2)
    """
    # Convert one-hot to sequence string
    data_str = ''.join([RNAConstants.IDX_TO_BASE[list(d).index(1)] for d in data])
    n = len(data_str)
    
    # Create sequence indices and pairing matrix
    seq_indices = torch.tensor([RNAConstants.BASE_TO_IDX[base] for base in data_str], device=device)
    pairing_strengths = _create_pairing_matrix(device)
    
    # Vectorized pairing matrix creation
    i_idx = seq_indices.unsqueeze(1).expand(n, n)
    j_idx = seq_indices.unsqueeze(0).expand(n, n)
    mat = pairing_strengths[i_idx, j_idx]
    
    # Pre-compute grid and weights
    i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
    
    # Compute m1 (backward direction)
    t_vals_m1 = torch.arange(30, device=device)
    exp_weights_m1 = torch.exp(-0.5 * t_vals_m1 * t_vals_m1)
    m1 = _compute_weighted_sum(mat, i, j, t_vals_m1, exp_weights_m1, -1, 1, n)
    
    # Compute m2 (forward direction)
    t_vals_m2 = torch.arange(1, 30, device=device)
    exp_weights_m2 = torch.exp(-0.5 * t_vals_m2 * t_vals_m2)
    m2 = _compute_weighted_sum(mat, i, j, t_vals_m2, exp_weights_m2, 1, -1, n)
    
    # Apply constraint: m2 = 0 where m1 = 0
    m2 = torch.where(m1 == 0, 0.0, m2)
    
    return m1 + m2


def constraint_matrix_batch_addnc(x: torch.Tensor) -> torch.Tensor:
    """
    Compute constraint matrix using optimized einsum operations.
    
    Args:
        x: Base probabilities tensor (batch_size, seq_len, 4)
        
    Returns:
        Constraint matrix (batch_size, seq_len, seq_len)
    """
    base_a, base_u, base_c, base_g = x.unbind(dim=-1)
    
    # Compute outer products using einsum
    outer_products = [
        torch.einsum('bi,bj->bij', base_a, base_u),
        torch.einsum('bi,bj->bij', base_c, base_g),
        torch.einsum('bi,bj->bij', base_u, base_g),
        torch.einsum('bi,bj->bij', base_a, base_c),
        torch.einsum('bi,bj->bij', base_a, base_g),
        torch.einsum('bi,bj->bij', base_u, base_c),
        torch.einsum('bi,bj->bij', base_a, base_a),
        torch.einsum('bi,bj->bij', base_u, base_u),
        torch.einsum('bi,bj->bij', base_c, base_c),
        torch.einsum('bi,bj->bij', base_g, base_g)
    ]
    
    # Combine symmetric and diagonal terms
    symmetric_sum = sum(op + op.transpose(-1, -2) for op in outer_products[:6])
    diagonal_sum = sum(outer_products[6:])
    
    return symmetric_sum + diagonal_sum


# ===============================
# POSTPROCESSING FUNCTIONS
# ===============================

def contact_a(a_hat: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Compute contact matrix with optimized operations."""
    a = a_hat.square()
    a = a.add(a.transpose(-1, -2)).mul_(0.5)
    return a.mul(m)


def soft_sign(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Numerically stable soft sign function."""
    return torch.sigmoid(2 * k * x)


def postprocess_new_nc(u: torch.Tensor, x: torch.Tensor, lr_min: float, lr_max: float,
                      num_itr: int, rho: float, with_l1: bool, s: float, decay: float,
                      device: torch.device) -> torch.Tensor:
    """
    Optimized postprocessing with GPU support and memory optimizations.
    
    Args:
        u: Utility matrix (batch_size, seq_len, seq_len)
        x: RNA sequence one-hot (batch_size, seq_len, 4)
        lr_min: Learning rate for minimization
        lr_max: Learning rate for maximization
        num_itr: Number of iterations
        rho: Sparsity coefficient
        with_l1: Whether to use L1 regularization
        s: Threshold parameter
        decay: learning rate decay
        device: Target device
        
    Returns:
        Final contact matrix
    """
    # Ensure tensors are on correct device
    u, x = u.to(device), x.to(device)
    
    # Pre-compute matrices
    m = constraint_matrix_batch_addnc(x).float()
    u_thresholded = soft_sign(u - s) * u
    
    # Initialize variables
    a_hat = torch.sigmoid(u) * soft_sign(u - s).detach()
    contact_sum = torch.sum(contact_a(a_hat, m), dim=-1)
    lmbd = F.relu(contact_sum - 1).detach()
    
    # Pre-allocate tensors
    grad_a = torch.empty_like(u)
    lmbd_expanded = torch.empty_like(u)
    
    # Optimization loop
    for t in range(num_itr):
        # Update contact sum and lambda
        contact_sum = torch.sum(contact_a(a_hat, m), dim=-1)
        lmbd_sign = soft_sign(contact_sum - 1)
        lmbd_expanded = (lmbd * lmbd_sign).unsqueeze(-1).expand_as(u)
        
        # Compute and apply gradients
        grad_a = lmbd_expanded - u_thresholded * 0.5
        grad = a_hat * m * (grad_a + grad_a.transpose(-1, -2))
        a_hat.sub_(grad, alpha=lr_min)
        
        # Optional L1 regularization
        if with_l1:
            a_hat = F.relu(a_hat.abs() - rho * lr_min)
        
        # Update lambda and decay learning rates
        lmbd.add_(F.relu(contact_sum - 1), alpha=lr_max)
        lr_min *= decay
        lr_max *= decay
    
    return contact_a(a_hat, m)
