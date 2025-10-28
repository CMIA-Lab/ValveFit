import jax.numpy as jnp
from jax import jit, vmap, jacfwd
import numpy as np


def refine_knotvector(knotvector, p):
    knots = jnp.unique(knotvector)
    mids = 0.5 * (knots[1:] + knots[:-1])
    refined_knotvector = jnp.concatenate(
        [
            knotvector[:p],
            jnp.unique(jnp.sort(jnp.concatenate([knots, mids]))),
            knotvector[-p:],
        ]
    )
    return refined_knotvector


def generate_parametric_coordinates(shape, endpoint=False, delta=1e-5):
    ndim = len(shape)
    pts = jnp.stack(
        jnp.meshgrid(
            *[jnp.linspace(0, 1, shape[dim], endpoint=endpoint) for dim in range(ndim)]
        ),
        axis=-1,
    ).reshape(-1, ndim)
    return pts


def grevilleAbscissae(fn_sh, degrees, knotvectors):
    ndim = len(fn_sh)
    CP = np.zeros((*fn_sh, ndim))

    for pt in np.ndindex(fn_sh):
        CP[pt] = jnp.array(
            [
                np.sum(knotvectors[dim][pt[dim] + 1 : pt[dim] + degrees[dim] + 1])
                / degrees[dim]
                for dim in range(ndim)
            ]
        )

    return jnp.array(CP)


@jit
def find_span_array(params, U, degree):
    n = len(U) - degree - 1
    indices = jnp.searchsorted(U, params, side="right") - 1
    indices = jnp.where(indices > n, n, indices)
    indices = jnp.where(params == U[n + 1], n, indices)
    return indices


@jit
def divisionbyzero(numerator, denominator):
    force_zero = jnp.logical_and(numerator == 0, denominator == 0)

    return jnp.where(force_zero, jnp.float32(0.0), numerator) / jnp.where(
        force_zero, jnp.float32(1.0), denominator
    )


def basis_fns(params, knotvector, degree):
    params1d = jnp.array(params)
    U = jnp.expand_dims(params1d, -1)
    knots = jnp.expand_dims(knotvector, 0)
    spans = find_span_array(params1d, knotvector, degree)
    K = jnp.where(
        knots == knotvector[-1], knotvector[-1] + jnp.finfo(U.dtype).eps, knots
    )
    t1 = U >= K[..., :-1]
    t2 = U < K[..., 1:]
    N = (t1 * t2) + 0.0
    for p in range(1, degree + 1):
        term1 = divisionbyzero(
            N[..., :-1] * (U - K[..., : -p - 1]), K[..., p:-1] - K[..., : -p - 1]
        )
        term2 = divisionbyzero(
            N[..., 1:] * (K[..., p + 1 :] - U), K[..., p + 1 :] - K[..., 1:-p]
        )
        N = term1 + term2

    span_ind = jnp.arange(degree + 1) + spans[:, jnp.newaxis] - degree
    row_indices = jnp.arange(span_ind.shape[0])[:, jnp.newaxis]
    non_zero_N = N[row_indices, span_ind]
    return non_zero_N


def basis_fns_vectorized(params, knotvector, degree):
    params1d = jnp.array(params)
    U = jnp.expand_dims(params1d, -1)
    knots = jnp.expand_dims(knotvector, 0)

    spans = find_span_array(params1d, knotvector, degree)

    K = jnp.where(
        knots == knotvector[-1], knotvector[-1] + jnp.finfo(U.dtype).eps, knots
    )

    t1 = U >= K[..., :-1]
    t2 = U < K[..., 1:]

    N = (t1 * t2) + 0.0

    for p in range(1, degree + 1):

        term1 = divisionbyzero(
            N[..., :-1] * (U - K[..., : -p - 1]), K[..., p:-1] - K[..., : -p - 1]
        )

        term2 = divisionbyzero(
            N[..., 1:] * (K[..., p + 1 :] - U), K[..., p + 1 :] - K[..., 1:-p]
        )

        N = term1 + term2

    span_ind = jnp.arange(degree + 1) + spans[:, np.newaxis] - degree

    row_indices = np.arange(span_ind.shape[0])[:, np.newaxis]

    non_zero_N = N[row_indices, span_ind]

    return non_zero_N
