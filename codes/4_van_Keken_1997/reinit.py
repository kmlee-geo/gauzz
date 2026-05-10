import cupy as cp
import numpy as np

def compute_sign_function_gpu(phi, epsilon=None):
    if epsilon is None:
        dx = 1.0 / phi.shape[1]
        dy = 1.0 / phi.shape[0]
        epsilon = 1.5 * min(dx, dy)
    return phi / cp.sqrt(phi**2 + epsilon**2)


def initialize_subcell_fix_cpu(phi_cpu, dx, dy):
    phi_fixed = np.zeros_like(phi_cpu)
    is_fixed  = np.zeros_like(phi_cpu, dtype=bool)

    c = phi_cpu[1:-1, 1:-1]                  # interior center values
    best_abs = np.full(c.shape, np.inf)
    best_val = np.zeros(c.shape)

    for phi_n, spacing in (
        (phi_cpu[1:-1, :-2], dx),             # left  (i-1)
        (phi_cpu[1:-1, 2: ], dx),             # right (i+1)
        (phi_cpu[:-2,  1:-1], dy),            # down  (j-1)
        (phi_cpu[2:,   1:-1], dy),            # up    (j+1)
    ):
        cross    = (phi_n * c) <= 0
        theta    = np.abs(c) / (np.abs(c) + np.abs(phi_n) + 1e-300)
        dist_abs = theta * spacing
        dist_val = np.sign(c) * dist_abs

        improve  = cross & (dist_abs < best_abs)
        best_abs = np.where(improve, dist_abs, best_abs)
        best_val = np.where(improve, dist_val, best_val)

    has_cross = np.isfinite(best_abs)
    phi_fixed[1:-1, 1:-1] = np.where(has_cross, best_val, 0.0)
    is_fixed [1:-1, 1:-1] = has_cross
    return phi_fixed, is_fixed

def initialize_subcell_fix_cpu_eno(phi_cpu, dx, dy, epsilon=1e-10):
    phi_fixed = np.zeros_like(phi_cpu)
    is_fixed  = np.zeros_like(phi_cpu, dtype=bool)

    c = phi_cpu[2:-2, 2:-2]                  # interior center values
    best_abs = np.full(c.shape, np.inf)
    best_val = np.zeros(c.shape)

    def _minmod(a, b):
        return np.where(a * b <= 0, 0.0, np.where(np.abs(a) <= np.abs(b), a, b))

    def _eno_dist(c, phi_n, d2_a, d2_b, h):
        """Return (dist_abs, dist_val) to the zero crossing toward phi_n."""
        cross      = (c * phi_n) <= 0
        phi_dd     = _minmod(d2_a, d2_b) / (h * h)
        lin_d      = np.abs(c) / (np.abs(c) + np.abs(phi_n) + 1e-300)  # ∈ (0,1)

        prod = c * phi_n
        D    = (phi_dd / 2.0 - (c + phi_n) / (h * h)) ** 2 - 4.0 * prod / h ** 4

        # Guard denominators: phi_dd=0 or D<=0 → linear fallback
        safe_dd   = np.where(np.abs(phi_dd) > epsilon, phi_dd, 1.0)
        safe_sqD  = np.sqrt(np.maximum(D, 0.0))
        safe_denom = safe_sqD * h * h * safe_dd  # may be 0 when D=0; use_eno guards it

        eno_d  = 0.5 - prod / (safe_denom + 1e-300)   # fraction of h
        eno_d  = np.clip(eno_d, 0.01, 0.99)

        use_eno   = cross & (np.abs(phi_dd) > epsilon) & (D > 0.0)
        frac      = np.where(use_eno, eno_d, lin_d)
        dist_abs  = frac * h
        dist_val  = np.sign(c) * dist_abs
        return cross, dist_abs, dist_val

    def _update(best_abs, best_val, cross, dist_abs, dist_val):
        improve  = cross & (dist_abs < best_abs)
        best_abs = np.where(improve, dist_abs, best_abs)
        best_val = np.where(improve, dist_val, best_val)
        return best_abs, best_val

    # right  (i+1): needs i-1, i, i+1, i+2
    phi_n = phi_cpu[2:-2, 3:-1]
    d2_a  = phi_cpu[2:-2, 1:-3] - 2*c        + phi_n
    d2_b  = c                   - 2*phi_n     + phi_cpu[2:-2, 4:]
    best_abs, best_val = _update(best_abs, best_val,
                                 *_eno_dist(c, phi_n, d2_a, d2_b, dx))

    # left   (i-1): needs i-2, i-1, i, i+1
    phi_n = phi_cpu[2:-2, 1:-3]
    d2_a  = phi_cpu[2:-2, :-4]  - 2*phi_n    + c
    d2_b  = phi_n               - 2*c        + phi_cpu[2:-2, 3:-1]
    best_abs, best_val = _update(best_abs, best_val,
                                 *_eno_dist(c, phi_n, d2_a, d2_b, dx))

    # up     (j+1): needs j-1, j, j+1, j+2
    phi_n = phi_cpu[3:-1, 2:-2]
    d2_a  = phi_cpu[1:-3, 2:-2] - 2*c        + phi_n
    d2_b  = c                   - 2*phi_n     + phi_cpu[4:, 2:-2]
    best_abs, best_val = _update(best_abs, best_val,
                                 *_eno_dist(c, phi_n, d2_a, d2_b, dy))

    # down   (j-1): needs j-2, j-1, j, j+1
    phi_n = phi_cpu[1:-3, 2:-2]
    d2_a  = phi_cpu[:-4,  2:-2] - 2*phi_n    + c
    d2_b  = phi_n               - 2*c        + phi_cpu[3:-1, 2:-2]
    best_abs, best_val = _update(best_abs, best_val,
                                 *_eno_dist(c, phi_n, d2_a, d2_b, dy))

    has_cross = np.isfinite(best_abs)
    phi_fixed[2:-2, 2:-2] = np.where(has_cross, best_val, 0.0)
    is_fixed [2:-2, 2:-2] = has_cross
    return phi_fixed, is_fixed

    
def compute_eno2_derivatives_gpu(phi, dx, dy):
    phi_pad = cp.pad(phi, 2, mode='edge')
    
    phi_im2 = phi_pad[2:-2, :-4]
    phi_im1 = phi_pad[2:-2, 1:-3]
    phi_i   = phi_pad[2:-2, 2:-2]
    phi_ip1 = phi_pad[2:-2, 3:-1]
    phi_ip2 = phi_pad[2:-2, 4:]
    
    v0 = (phi_im1 - phi_im2) / dx
    v1 = (phi_i - phi_im1) / dx
    v2 = (phi_ip1 - phi_i) / dx
    v3 = (phi_ip2 - phi_ip1) / dx
    
    s0 = v1 - v0
    s1 = v2 - v1
    s2 = v3 - v2
    
    Dx_minus = cp.where(cp.abs(s0) <= cp.abs(s1), v1 - 0.5*s0, v1 + 0.5*s1)
    Dx_plus = cp.where(cp.abs(s1) <= cp.abs(s2), v2 - 0.5*s1, v2 + 0.5*s2)
    
    phi_jm2 = phi_pad[:-4, 2:-2]
    phi_jm1 = phi_pad[1:-3, 2:-2]
    phi_j = phi_pad[2:-2, 2:-2]
    phi_jp1 = phi_pad[3:-1, 2:-2]
    phi_jp2 = phi_pad[4:, 2:-2]
    
    v0 = (phi_jm1 - phi_jm2) / dy
    v1 = (phi_j - phi_jm1) / dy
    v2 = (phi_jp1 - phi_j) / dy
    v3 = (phi_jp2 - phi_jp1) / dy
    
    s0 = v1 - v0
    s1 = v2 - v1
    s2 = v3 - v2
    
    Dy_minus = cp.where(cp.abs(s0) <= cp.abs(s1), v1 - 0.5*s0, v1 + 0.5*s1)
    Dy_plus = cp.where(cp.abs(s1) <= cp.abs(s2), v2 - 0.5*s1, v2 + 0.5*s2)
    
    return Dx_minus, Dx_plus, Dy_minus, Dy_plus


def godunov_hamiltonian_gpu(Dx_minus, Dx_plus, Dy_minus, Dy_plus, sign_phi):
    ax_pos = cp.maximum(Dx_minus, 0)**2
    bx_pos = cp.minimum(Dx_plus, 0)**2
    ay_pos = cp.maximum(Dy_minus, 0)**2
    by_pos = cp.minimum(Dy_plus, 0)**2
    
    ax_neg = cp.minimum(Dx_minus, 0)**2
    bx_neg = cp.maximum(Dx_plus, 0)**2
    ay_neg = cp.minimum(Dy_minus, 0)**2
    by_neg = cp.maximum(Dy_plus, 0)**2
    
    H_pos = cp.sqrt(cp.maximum(ax_pos, bx_pos) + cp.maximum(ay_pos, by_pos)) - 1.0
    H_neg = cp.sqrt(cp.maximum(ax_neg, bx_neg) + cp.maximum(ay_neg, by_neg)) - 1.0
    
    return cp.where(sign_phi > 0, H_pos, H_neg)


def red_black_update_gpu(phi, phi_fixed, is_fixed, sign_phi, dx, dy, dt, sweep_dir):
    ny, nx = phi.shape
    i_idx = cp.arange(nx)
    j_idx = cp.arange(ny)[:, None]
    
    if sweep_dir == 0:
        red_mask = ((i_idx + j_idx) % 2 == 0)
    elif sweep_dir == 1:
        red_mask = ((nx - 1 - i_idx + j_idx) % 2 == 0)
    elif sweep_dir == 2:
        red_mask = ((i_idx + ny - 1 - j_idx) % 2 == 0)
    else:
        red_mask = ((nx - 1 - i_idx + ny - 1 - j_idx) % 2 == 0)
    
    black_mask = ~red_mask
    
    phi = update_points_rk2_gpu(phi, red_mask, phi_fixed, is_fixed, sign_phi, dx, dy, dt)
    phi = update_points_rk2_gpu(phi, black_mask, phi_fixed, is_fixed, sign_phi, dx, dy, dt)
    
    return phi


def update_points_rk2_gpu(phi, mask, phi_fixed, is_fixed, sign_phi, dx, dy, dt):
    Dx_m, Dx_p, Dy_m, Dy_p = compute_eno2_derivatives_gpu(phi, dx, dy)
    H1 = godunov_hamiltonian_gpu(Dx_m, Dx_p, Dy_m, Dy_p, sign_phi)
    
    phi_star = phi - dt * sign_phi * H1
    phi_star = cp.where(mask & ~is_fixed, phi_star, phi)
    
    Dx_m, Dx_p, Dy_m, Dy_p = compute_eno2_derivatives_gpu(phi_star, dx, dy)
    H2 = godunov_hamiltonian_gpu(Dx_m, Dx_p, Dy_m, Dy_p, sign_phi)
    
    phi_new = phi - 0.5 * dt * sign_phi * (H1 + H2)
    
    phi_result = cp.where(mask & ~is_fixed, phi_new, phi)
    phi_result = cp.where(is_fixed, phi_fixed, phi_result)
    
    phi_result = cp.where((mask & ~is_fixed & (sign_phi > 0)), 
                          cp.maximum(phi_result, 0), phi_result)
    phi_result = cp.where((mask & ~is_fixed & (sign_phi < 0)), 
                          cp.minimum(phi_result, 0), phi_result)
    
    return phi_result


def compute_gradient_error_gpu(phi, dx, dy, is_fixed):
    grad_y, grad_x = cp.gradient(phi, dy, dx, edge_order=2)
    grad_mag = cp.sqrt(grad_x**2 + grad_y**2)
    
    error = cp.abs(grad_mag - 1.0)
    error = cp.where(is_fixed, 0, error)
    
    return float(cp.mean(error)), float(cp.max(error))


def min2010_reinitialize_gpu(phi_gpu, dx, dy, max_iter=15, tol=1e-5, verbose=False):
    phi_cpu = cp.asnumpy(phi_gpu)
    phi_fixed_cpu, is_fixed_cpu = initialize_subcell_fix_cpu(phi_cpu, dx, dy)
    
    phi_fixed = cp.asarray(phi_fixed_cpu)
    is_fixed = cp.asarray(is_fixed_cpu)
    sign_phi = compute_sign_function_gpu(phi_gpu)
    
    dt = 0.5 * min(dx, dy)
    phi_current = phi_gpu.copy()
    errors = []
    
    for iteration in range(max_iter):
        for sweep_dir in range(4):
            phi_current = red_black_update_gpu(
                phi_current, phi_fixed, is_fixed, 
                sign_phi, dx, dy, dt, sweep_dir
            )
        
        error_mean, error_max = compute_gradient_error_gpu(phi_current, dx, dy, is_fixed)
        errors.append(error_mean)
        
        if verbose:
            print(f"Iteration {iteration+1:3d}: Mean |∇φ|-1 = {error_mean:.6f}, Max = {error_max:.6f}")
        
        if error_mean < tol:
            if verbose:
                print(f"Converged at iteration {iteration+1}")
            break
    
    return phi_current, errors