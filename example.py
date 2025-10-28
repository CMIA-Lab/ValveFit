# %%
import os
import pickle
import numpy as np
import pyiges
import jax.numpy as jnp
from jax import value_and_grad, jit, random
from tqdm import tqdm
import optax
from functools import partial
from copy import deepcopy

from ValveFit.spline_space import BSpline, PeriodicBSpline, HeartValve
from ValveFit.periodic_splines_funcs import (
    compute_tensor_product,
    evaluate,
    linear_transform_valve_non_uniform_scaling,
    periodic_ctrl_pts,
    bspline_jacobian,
)
from ValveFit.bspline_funcs import (
    generate_parametric_coordinates,
    find_span_array,
)
from ValveFit.io import export_quad_mesh_vtk, save_pc_as_vtk
from ValveFit.plot import plot_log_loss
from ValveFit.regularization_terms import (
    compute_tangents_normals,
    compute_tangent_dot,
    compute_tangent_point_energy,
)
from ValveFit.shape_fidelity_terms import (
    chamfer_distance,
    compute_surface_area,
)

# %%

W_CD = 80
W_HD = 30
W_ORTH = 5
W_NORM = 2
W_TPE = 1e-7
ALPHA = 4.0

MAX_ITER = 20000
LEARNING_RATE = 0.001
TIMESTEP = 30

OUTPUT_DIR = "results/example_single_timestep"

# %%

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

weights = {
    "W_CD": W_CD,
    "W_HD": W_HD,
    "W_ORTH": W_ORTH,
    "W_NORM": W_NORM,
    "W_TPE": W_TPE,
    "ALPHA": ALPHA,
    "MAX_ITER": MAX_ITER,
    "LEARNING_RATE": LEARNING_RATE,
    "TIMESTEP": TIMESTEP,
}

with open(OUTPUT_DIR + "/parameters.txt", "w") as f:
    f.write(f"Hyperparameters:\n")
    for key, value in weights.items():
        f.write(f"{key}: {value}\n")

# %%


def get_CP_indices_v2(bsplines, params):
    spans = jnp.stack(
        [
            find_span_array(params[:, dim], bspline.knotvector, bspline.degree)
            for dim, bspline in enumerate(bsplines)
        ]
    ).T
    p, q = tuple(bspline.degree for bspline in bsplines)
    x_offsets, y_offsets = jnp.meshgrid(
        jnp.arange(p + 1), jnp.arange(q + 1), indexing="ij"
    )
    x_indices = spans[:, 0, jnp.newaxis, jnp.newaxis] + x_offsets - p
    y_indices = spans[:, 1, jnp.newaxis, jnp.newaxis] + y_offsets - q

    n_unique_ctrl_x = len(bsplines[0].knotvector) - 2 * p - 1
    x_indices = x_indices % n_unique_ctrl_x

    return [x_indices, y_indices]


# %%

template = pyiges.read("data/Leaflet_Ajith.igs")
bspline_surf = template.bspline_surfaces()[0].to_geomdl()

# Set evaluation resolution
nevalu, nevalv = (100, 20)  # Circumferential x Radial
bspline_surf.delta_u = 1 / nevalu
bspline_surf.delta_v = 1 / nevalv
bspline_surf.evaluate()

# Get template surface points
surf = jnp.array(bspline_surf.evalpts).reshape(nevalu, nevalv, 3).transpose(1, 0, 2)
eval_sh = (nevalu, nevalv)
nevalu_fine, nevalv_fine = (100, 20)
eval_sh_fine = (nevalu_fine, nevalv_fine)

# Export template mesh
export_quad_mesh_vtk(
    np.array(surf),
    OUTPUT_DIR + "/template_mesh.vtk",
)

# %%

# Define knot vectors
kv5 = jnp.concatenate(
    [
        jnp.array([-0.15, -0.1, -0.05]),
        jnp.linspace(0, 1, 31, endpoint=True),
        jnp.array([1.05, 1.1, 1.15]),
    ]
)
kv7 = jnp.array([0, 0, 0, 0, 0.33, 0.66, 1, 1, 1, 1])

bs1 = BSpline(knotvector=kv7, degree=3)
bs2 = PeriodicBSpline(knotvector=kv5, degree=3)

valve = HeartValve([bs2, bs1])
degrees = valve.degrees
knotvectors = valve.knotvectors


# %%

params = generate_parametric_coordinates(eval_sh)
params_fine = generate_parametric_coordinates(eval_sh_fine)

fns = compute_tensor_product(params, knotvectors, degrees)
CP_indices = get_CP_indices_v2(valve.bsplines, params)

fns_fine = compute_tensor_product(params_fine, knotvectors, degrees)
CP_indices_fine = get_CP_indices_v2(valve.bsplines, params_fine)

ctrl_pts = random.uniform(random.PRNGKey(0), (valve.sh_fns[0], valve.sh_fns[1], 3))

# %%

MSE = lambda x, y: jnp.mean((x - y) ** 2)


def compute_MSE_loss(ctrl_pts, target):
    pts = evaluate(fns, ctrl_pts, CP_indices).reshape(nevalv, nevalu, 3)
    target = target.reshape(nevalv, nevalu, 3)
    loss = MSE(pts, target)
    return loss


loss_grad = jit(value_and_grad(compute_MSE_loss, argnums=0))

pbar = tqdm(range(1000))

solver = optax.adam(learning_rate=0.1)
opt_state = solver.init(ctrl_pts)

for i in pbar:
    value, grads = loss_grad(ctrl_pts, target=surf)
    updates, opt_state = solver.update(grads, opt_state)
    ctrl_pts = optax.apply_updates(ctrl_pts, updates)
    pbar.set_description(f"MSE: {value:.6f}")

pts = np.array(evaluate(fns_fine, ctrl_pts, CP_indices_fine))
t1, t2, n = compute_tangents_normals(
    params_fine, ctrl_pts, knotvectors, degrees, CP_indices_fine, unit_vectors=True
)

export_quad_mesh_vtk(
    pts.reshape(nevalv_fine, nevalu_fine, 3),
    OUTPUT_DIR + "/template_with_normals.vtk",
    normals=np.array(n),
    valve=valve,
)


# %%

point_cloud = jnp.array(np.loadtxt(f"data/step{TIMESTEP}_1200.xyz"))


def compute_loss_terms(ctrl_pts, params, fns, knotvectors, degrees, CP_indices, target):
    pts = evaluate(fns, ctrl_pts, CP_indices)
    diff = pts[:, None, :] - target[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2 + 1e-8, axis=-1))

    NN_s = jnp.min(dists, axis=1)
    NN_t = jnp.min(dists, axis=0)
    D_cd = NN_t.mean()

    D_hd = jnp.array([NN_t.max(), NN_s.max()]).max()

    pts_fine = evaluate(fns_fine, ctrl_pts, CP_indices_fine)
    jac = bspline_jacobian(params_fine, ctrl_pts, knotvectors, degrees, CP_indices_fine)
    t1 = jac[:, :, 0]
    t2 = jac[:, :, 1]
    t1_norm = jnp.linalg.norm(t1, axis=-1, keepdims=True) + 1e-8
    t2_norm = jnp.linalg.norm(t2, axis=-1, keepdims=True) + 1e-8
    t1_unit = t1 / t1_norm
    t2_unit = t2 / t2_norm
    normals = jnp.cross(t1, t2)

    R_t_dot = compute_tangent_dot(t1_unit, t2_unit)

    normals_normalized = (normals - normals.min()) / (normals.max() - normals.min())
    R_n_var = jnp.abs(
        normals_normalized[:, -1] - normals_normalized[:, -1].mean()
    ).max()

    tpe_mean, tpe_values = compute_tangent_point_energy(normals, pts_fine, alpha=ALPHA)
    R_tpe = tpe_values.max()

    return jnp.array(
        [
            W_CD * D_cd,
            W_HD * D_hd,
            W_ORTH * R_t_dot,
            W_NORM * R_n_var,
            W_TPE * R_tpe,
        ]
    )


def compute_loss(ctrl_pts, params, fns, knotvectors, degrees, CP_indices, target):
    return compute_loss_terms(
        ctrl_pts, params, fns, knotvectors, degrees, CP_indices, target
    ).sum()


loss_grad = jit(value_and_grad(compute_loss, argnums=0), static_argnums=(4,))


@partial(jit, static_argnums=(3))
def update_ctrl_pts(ctrl_pts, target, opt_state, solver):
    value, grads = loss_grad(
        ctrl_pts, params, fns, knotvectors, degrees, CP_indices, target=target
    )
    updates, opt_state = solver.update(grads, opt_state)
    ctrl_pts = optax.apply_updates(ctrl_pts, updates)
    return ctrl_pts, opt_state, value


# %%

ctrl_pts_aligned, _ = linear_transform_valve_non_uniform_scaling(
    valve, ctrl_pts, point_cloud, n_iter=2000
)

pts_aligned = np.array(evaluate(fns_fine, ctrl_pts_aligned, CP_indices_fine))
t1, t2, normals = compute_tangents_normals(
    params_fine,
    ctrl_pts_aligned,
    knotvectors,
    degrees,
    CP_indices_fine,
    unit_vectors=True,
)

export_quad_mesh_vtk(
    pts_aligned.reshape(nevalv_fine, nevalu_fine, 3),
    OUTPUT_DIR + "/transformed_template.vtk",
    normals=normals.reshape(*eval_sh_fine, 3),
    valve=valve,
)

loss_terms_init = compute_loss_terms(
    ctrl_pts_aligned, params, fns, knotvectors, degrees, CP_indices, point_cloud
)

# %%

solver = optax.adam(learning_rate=LEARNING_RATE)
opt_state = solver.init(ctrl_pts_aligned)
loss_history = []

pbar = tqdm(range(MAX_ITER))
CP = ctrl_pts_aligned

for i in pbar:
    CP, opt_state, value = update_ctrl_pts(CP, point_cloud, opt_state, solver)
    if i % 100 == 0:
        pbar.set_description(f"Loss: {value:.6f}")
        loss_history.append(value)

# %%

t1, t2, normals = compute_tangents_normals(
    params_fine, CP, knotvectors, degrees, CP_indices_fine, unit_vectors=True
)
pts = np.array(evaluate(fns_fine, CP, CP_indices_fine))

loss_terms_final = compute_loss_terms(
    CP, params, fns, knotvectors, degrees, CP_indices, point_cloud
)

_, nnd = chamfer_distance(
    point_cloud, jnp.array(pts), one_sided=0, return_distances=True
)
snnd = nnd / jnp.sqrt(compute_surface_area(pts.reshape(*eval_sh_fine, 3)))

# %%
save_pc_as_vtk(
    point_cloud,
    OUTPUT_DIR + f"/sNND_{TIMESTEP}.vtk",
    metadata=snnd,
    metadata_label="sNND",
)

export_quad_mesh_vtk(
    pts.reshape(nevalv_fine, nevalu_fine, 3),
    tangent_vectors=(t1, t2),
    normals=normals,
    filename=OUTPUT_DIR + f"/fitted_surface_{TIMESTEP}.vtk",
    valve=valve,
)

np.savetxt(OUTPUT_DIR + f"/{TIMESTEP}.xyz", pts)

plot_log_loss(loss_history, logscale=True, filename=OUTPUT_DIR + f"/loss_{TIMESTEP}")

data = {
    "control_points": CP,
    "surface_points": pts,
    "sNND": snnd,
    "loss_history": loss_history,
    "loss_terms_initial": loss_terms_init,
    "loss_terms_final": loss_terms_final,
    "knotvectors": knotvectors,
    "degrees": degrees,
    "weights": weights,
}

with open(OUTPUT_DIR + "/results.pkl", "wb") as f:
    pickle.dump(data, f)
