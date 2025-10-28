import numpy as np
import jax.numpy as jnp
import pyvista as pv
import vtk
from .bspline_funcs import generate_parametric_coordinates


def save_pc_as_vtk(pc, filename, metadata=None, metadata_label=None):
    points = np.asarray(pc)
    poly = pv.PolyData(points)
    if metadata is not None:
        if metadata_label is None:
            metadata_label = "metadata"
        poly[metadata_label] = np.asarray(metadata)
    poly.save(filename)


def export_quad_mesh_vtk(
    points,
    filename,
    fitting_error=None,
    tangent_vectors=None,
    normals=None,
    valve=None,
    TPE=None,
):
    n, m, _ = points.shape

    vertices = points.reshape(-1, 3)

    quads = []
    for i in range(n - 1):
        for j in range(m):
            j_plus_1_wrapped = (j + 1) % m
            v1_idx = i * m + j
            v2_idx = i * m + j_plus_1_wrapped
            v3_idx = (i + 1) * m + j_plus_1_wrapped
            v4_idx = (i + 1) * m + j

            quads.append([v1_idx, v2_idx, v3_idx, v4_idx])

    faces = np.hstack([[4] + quad for quad in quads])
    mesh = pv.PolyData(vertices, faces)

    if tangent_vectors is not None:
        tangents_u = np.array(tangent_vectors[0]).reshape(-1, 3)
        tangents_v = np.array(tangent_vectors[1]).reshape(-1, 3)
        mesh["tangent_u"] = tangents_u
        mesh["tangent_v"] = tangents_v

    if normals is not None:
        normals = np.array(normals).reshape(n, m, -1).reshape(-1, 3)
        mesh["normals"] = normals

    if fitting_error is not None:
        mesh["sNND"] = np.array(fitting_error)

    if TPE is not None:
        TPE_array = np.array(TPE).reshape(-1)
        mesh["TPE"] = TPE_array

    mesh.save(filename)

    if valve is not None:
        plot_valve_knotlines_from_surface(
            valve=valve,
            surface_points=points,
            filename=filename[:-4] + "_lines.vtk",
        )


def plot_valve_knotlines_from_surface(
    valve,
    surface_points,
    filename=None,
):
    degrees = valve.degrees
    knotvectors = valve.knotvectors

    n, m, _ = surface_points.shape
    p, q = degrees
    u_knots = knotvectors[0][p:-p]
    v_knots = knotvectors[1][q:-q]

    lines = []

    u_params = np.linspace(0, 1, m, endpoint=False)
    v_params = np.linspace(0, 1, n, endpoint=False)

    for knot in u_knots[:-1]:
        u_idx = np.argmin(np.abs(u_params - knot))
        knot_line_points = surface_points[:, u_idx, :]
        lines.append(np.array(knot_line_points))

    for knot in v_knots:
        v_idx = np.argmin(np.abs(v_params - knot))
        knot_line_points = surface_points[v_idx, :, :]
        periodic_line_points = np.vstack([knot_line_points, knot_line_points[0:1, :]])
        lines.append(np.array(periodic_line_points))

    if filename is not None:
        save_lines_to_vtk(lines, filename)


def save_lines_to_vtk(lines_points_list, filename="lines.vtk"):
    vtk_points = vtk.vtkPoints()
    point_map = {}
    point_id_counter = 0

    all_points = [point for line in lines_points_list for point in line]

    for point in all_points:
        point_tuple = tuple(point)
        if point_tuple not in point_map:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
            point_map[point_tuple] = point_id_counter
            point_id_counter += 1

    vtk_lines = vtk.vtkCellArray()

    for line_points in lines_points_list:
        num_points_in_line = len(line_points)

        poly_line = vtk.vtkPolyLine()
        poly_line.GetPointIds().SetNumberOfIds(num_points_in_line)

        for i, point in enumerate(line_points):
            point_tuple = tuple(point)
            point_id = point_map[point_tuple]
            poly_line.GetPointIds().SetId(i, point_id)

        vtk_lines.InsertNextCell(poly_line)

    poly_data = vtk.vtkPolyData()

    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly_data)

    writer.Write()
