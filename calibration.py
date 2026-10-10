"""Equal-subject validation NMSE calibration of four convex weights."""
import numpy as np
import torch
from scipy.optimize import minimize

@torch.no_grad()
def quadratic(experts, z, val, hidden, subjects):
    a = torch.zeros(4, 4, device='cuda', dtype=torch.float64)
    b = torch.zeros(4, device='cuda', dtype=torch.float64)
    c = torch.zeros((), device='cuda', dtype=torch.float64)
    group = subjects[val]
    unique = np.unique(group)
    counts = {s: np.sum(group == s) for s in unique}
    for start in range(0, len(val), 64):
        ids = val[start:start + 64]
        y = z[torch.as_tensor(ids, device='cuda')]
        h = torch.as_tensor(hidden[start:start + 64], device='cuda')
        m = (~h).float().unsqueeze(-1).expand_as(y)
        p = torch.stack([model(y * m, m).double() for model in experts], 1)
        yd = y.double()
        energy = yd.square().sum(-1).clamp_min(1e-12)
        factor = h / energy / h.sum(1)[:, None]
        sw = torch.tensor([1 / len(unique) / counts[s] for s in subjects[ids]], device='cuda')
        factor = factor * sw[:, None]
        a += torch.einsum('bict,bjct,bc->ij', p, p, factor)
        b += torch.einsum('bict,bct,bc->i', p, yd, factor)
        c += (yd.square().sum(-1) * factor).sum()
    return (a.cpu().numpy(), b.cpu().numpy(), float(c))

def solve(a, b, c):
    result = minimize(lambda w: float(w @ a @ w - 2 * b @ w + c), np.ones(4) / 4, jac=lambda w: 2 * (a @ w - b), bounds=[(0, 1)] * 4, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(4)}], method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000})
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.clip(result.x, 0, 1)
    weights /= weights.sum()
    return (weights, float(weights @ a @ weights - 2 * b @ weights + c))
