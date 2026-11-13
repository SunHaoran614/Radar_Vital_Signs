"""Halir--Flusser algebraic ellipse calibration, with explicit numeric fallback."""
import numpy as np


def fit(z):
    z = np.asarray(z, np.complex128)
    if z.ndim != 1 or len(z) < 6 or not np.isfinite(z).all():
        raise ValueError('Expected finite complex vector of at least six samples')
    mean = z.mean()
    scale = float(np.sqrt(np.mean(abs(z - mean) ** 2)))
    safe_scale = scale if scale > 0 else 1.
    u = (z - mean) / safe_scale
    x, y = u.real, u.imag
    linear = np.column_stack((x, y, np.ones(len(x))))
    singular = np.linalg.svd(linear, compute_uv=False)
    condition = float(singular[-1] / singular[0])
    base = dict(mean=complex(mean), scale=safe_scale, condition=condition, center=np.zeros(2), operator=np.eye(2), coefficients=np.array([1., 0., 1., 0., 0., -1.]), axis_ratio=1., fallback='')
    if scale == 0 or condition < 1e-6:
        base['fallback'] = 'zero' if scale == 0 else 'linear'
        return base
    quadratic = np.column_stack((x*x, x*y, y*y))
    s1 = quadratic.T @ quadratic
    s2 = quadratic.T @ linear
    s3 = linear.T @ linear
    tail = -np.linalg.solve(s3, s2.T)
    reduced = s1 + s2 @ tail
    reduced = np.stack((reduced[2] / 2, -reduced[1], reduced[0] / 2))
    _, vectors = np.linalg.eig(reduced)
    candidates = []
    for vector in vectors.T:
        if abs(vector.imag).max() > 1e-8:
            continue
        v = vector.real
        constraint = 4*v[0]*v[2] - v[1]**2
        if constraint > 0:
            coefficients = np.r_[v, tail @ v] / np.sqrt(constraint)
            candidates.append(coefficients)
    if len(candidates) != 1:
        base['fallback'] = 'candidate'
        return base
    coefficients = candidates[0]
    if coefficients[0] + coefficients[2] < 0:
        coefficients = -coefficients
    a, b, c, d, e, f = coefficients
    q = np.array([[a, b/2], [b/2, c]])
    values, vectors = np.linalg.eigh(q)
    if values[0] <= 1e-12 * values[-1]:
        base['fallback'] = 'not_spd'
        return base
    center = -np.linalg.solve(q, np.array([d, e])) / 2
    level = float(center @ q @ center - f)
    if level <= 1e-12:
        base['fallback'] = 'level'
        return base
    operator = (vectors * np.sqrt(values / level)[None]) @ vectors.T
    base.update(center=center, operator=operator, coefficients=coefficients, axis_ratio=float(np.sqrt(values[-1] / values[0])))
    return base


def apply(z, calibration):
    u = (np.asarray(z) - calibration['mean']) / calibration['scale']
    xy = np.stack((u.real, u.imag)) - calibration['center'][:, None]
    corrected = calibration['operator'] @ xy
    return corrected[0] + 1j*corrected[1]
