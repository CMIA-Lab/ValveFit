import jax.numpy as jnp


def _pairwise_distances(x, y):
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))


def chamfer_distance(x, y, one_sided=None, return_distances=False):
    dists = _pairwise_distances(x, y)

    if one_sided is not None:
        if one_sided == 0:
            x_closest_points = jnp.min(dists, axis=1)
            if return_distances:
                return jnp.mean(x_closest_points), x_closest_points
            return jnp.mean(x_closest_points)
        elif one_sided == 1:
            y_closest_points = jnp.min(dists, axis=0)
            if return_distances:
                return jnp.mean(y_closest_points), y_closest_points
            return jnp.mean(y_closest_points)
        else:
            raise ValueError("one_sided parameter must be None, 0, or 1")

    y_closest_points = jnp.min(dists, axis=0)
    x_closest_points = jnp.min(dists, axis=1)

    if return_distances:
        return (
            jnp.mean(x_closest_points) + jnp.mean(y_closest_points),
            x_closest_points,
            y_closest_points,
        )

    return jnp.mean(x_closest_points) + jnp.mean(y_closest_points)


def hausdorff_distance(x, y, return_distances=False):
    dists = _pairwise_distances(x, y)

    min_dists_x_to_y = jnp.min(dists, axis=1)
    min_dists_y_to_x = jnp.min(dists, axis=0)

    hausdorff_dist = jnp.max(
        jnp.maximum(jnp.max(min_dists_x_to_y), jnp.max(min_dists_y_to_x))
    )

    if return_distances:
        return hausdorff_dist, jnp.max(min_dists_x_to_y), jnp.max(min_dists_y_to_x)

    return hausdorff_dist


def compute_surface_area(points):
    nx, ny, _ = points.shape
    edge = points[0, :, :]
    points = jnp.vstack([points, edge.reshape(1, ny, 3)])

    nx, ny, _ = points.shape

    i_indices, j_indices = jnp.arange(nx - 1), jnp.arange(ny - 1)
    I, J = jnp.meshgrid(i_indices, j_indices, indexing="ij")

    I_flat, J_flat = I.flatten(), J.flatten()

    p00 = points[I_flat, J_flat]  # (i, j)
    p10 = points[I_flat + 1, J_flat]  # (i+1, j)
    p01 = points[I_flat, J_flat + 1]  # (i, j+1)
    p11 = points[I_flat + 1, J_flat + 1]  # (i+1, j+1)

    v1_t1 = p10 - p00
    v2_t1 = p01 - p00

    v1_t2 = p11 - p10
    v2_t2 = p01 - p10

    cross_t1 = jnp.cross(v1_t1, v2_t1)
    cross_t2 = jnp.cross(v1_t2, v2_t2)

    areas_t1 = 0.5 * jnp.linalg.norm(cross_t1 + 1e-10, axis=1)
    areas_t2 = 0.5 * jnp.linalg.norm(cross_t2 + 1e-10, axis=1)

    total_area = jnp.sum(areas_t1) + jnp.sum(areas_t2)

    return total_area
