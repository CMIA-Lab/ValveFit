import jax.numpy as jnp
from jax import vmap
import numpy as np
from ValveFit.periodic_splines_funcs import bspline_jacobian


def uniform_between_min_max(ctrl_pts):
    n, m, _ = ctrl_pts.shape
    min_points = ctrl_pts.min(axis=0)
    max_points = ctrl_pts.max(axis=0)
    alphas = jnp.linspace(0, 1, n).reshape(n, 1, 1)
    new_ctrl_pts = (1 - alphas) * min_points + alphas * max_points
    return new_ctrl_pts


def compute_tangents_normals(
    params, ctrl_pts, knotvectors, degrees, CP_indices, unit_vectors=False
):
    jac = bspline_jacobian(params, ctrl_pts, knotvectors, degrees, CP_indices)
    t1 = jac[:, :, 0]
    t2 = jac[:, :, 1]
    t1_norm = jnp.linalg.norm(t1, axis=-1, keepdims=True) + 1e-8
    t2_norm = jnp.linalg.norm(t2, axis=-1, keepdims=True) + 1e-8
    t1_unit = t1 / t1_norm
    t2_unit = t2 / t2_norm
    normals = jnp.cross(t1_unit, t2_unit)
    if unit_vectors:
        return t1_unit, t2_unit, normals
    return t1, t2, normals


def compute_tangent_dot(t1, t2):
    return jnp.mean(jnp.abs(vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(t1, t2)))


def compute_z_normal_var(normals):
    return jnp.mean(jnp.abs(jnp.var(normals[:, -1], axis=0)))


def compute_z_normal_max_deviation(normals):
    return jnp.abs(normals[:, -1] - normals[:, -1].mean()).max()


def compute_z_normal_max_deviation_v2(normals):
    normals = (normals - normals.min()) / (normals.max() - normals.min())
    return jnp.abs(normals[:, -1] - normals[:, -1].mean()).max()


def compute_squared_tangent_lengths(t1, t2):
    R_t1_norm = jnp.mean(t1**2)
    R_t2_norm = jnp.mean(t2**2)

    R_t_norm = 0.5 * (R_t1_norm + R_t2_norm)
    return R_t_norm


def compute_tangent_point_energy(unit_normals, pts, alpha=2.0):
    diff = pts[:, None, :] - pts[None, :, :]
    numerator = (jnp.abs(jnp.sum(unit_normals[:, None, :] * diff, axis=-1))) ** alpha
    sum_sq_diff = jnp.sum(diff**2, axis=-1)
    denominator = jnp.abs(sum_sq_diff + 1e-8) ** alpha
    term_values = jnp.where(sum_sq_diff == 0, 0, numerator / (denominator + 1e-8))
    return term_values.mean(), term_values
