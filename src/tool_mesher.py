import sys
from pathlib import Path
from typing import Dict, List

import gmsh

import config
from utils import generate_airfoil_contour


def _mesh_file() -> Path:
    return config.work_dir / "sims" / "airfoil.msh"


def _bl_thickness() -> float:
    """Total BL thickness from first height, growth ratio, and layer count."""
    h0 = config.bl_first_height
    r = config.bl_ratio
    n = config.bl_layers
    if abs(r - 1.0) < 1e-12:
        return h0 * n
    return h0 * (r**n - 1.0) / (r - 1.0)


def _setup_boundary_layer(airfoil_curves: List[int], te_pt: int) -> int:
    """Create Gmsh BoundaryLayer field aligned with naca_boundary_layer_2d.py."""
    bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(bl_field, "CurvesList", airfoil_curves)
    gmsh.model.mesh.field.setNumber(bl_field, "Size", config.bl_first_height)
    gmsh.model.mesh.field.setNumber(bl_field, "Ratio", config.bl_ratio)
    gmsh.model.mesh.field.setNumber(bl_field, "Quads", 1)
    gmsh.model.mesh.field.setNumber(bl_field, "Thickness", _bl_thickness())
    gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", 7)
    gmsh.model.mesh.field.setNumbers(bl_field, "FanPointsList", [te_pt])
    gmsh.model.mesh.field.setAsBoundaryLayer(bl_field)
    return bl_field


def _classify_lateral_surface(
    dim: int,
    tag: int,
    chord: float,
    z_tol: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    """Classify extruded side faces as airfoil wall or farfield sub-patch."""
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    if abs(zmax - zmin) <= z_tol:
        return "cap"

    x_pad = 0.05 * chord
    in_chord = xmin >= -x_pad and xmax <= chord + x_pad
    narrow_y = (ymax - ymin) <= 0.6 * chord
    if in_chord and narrow_y:
        return "airfoil"

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    tol_x = 0.02 * (x_max - x_min)
    tol_y = 0.02 * (y_max - y_min)
    if cx <= x_min + tol_x:
        return "inlet"
    if cx >= x_max - tol_x:
        return "outlet"
    if cy <= y_min + tol_y:
        return "bottom"
    return "top"


def mesher(coeffs: Dict) -> bool:
    """
    从 CST 翼型点列生成绕流网格（含 Z 向 1 层挤出），保存为 MSH。
    """
    x_c, y_c = generate_airfoil_contour(coeffs)
    chord = config.chord
    r_far = config.farfield_radius * chord

    # Drop duplicate TE point so OCC spline stays non-degenerate.
    if len(x_c) > 1 and abs(x_c[0] - x_c[-1]) < 1e-10 and abs(y_c[0] - y_c[-1]) < 1e-10:
        x_c = x_c[:-1]
        y_c = y_c[:-1]

    x_min = -0.5 * chord - r_far
    x_max = chord + r_far
    y_min = -r_far
    y_max = r_far

    h_wall = config.wall_mesh_size * chord
    h_far = chord / 8.0
    dz = 0.1 * chord
    z_tol = max(1e-9, 1e-6 * dz)

    initialized = False
    try:
        gmsh.initialize()
        initialized = True
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("airfoil")

        # --- airfoil profile (OCC) ---
        airfoil_pts: List[int] = []
        for i in range(len(x_c)):
            tag = gmsh.model.occ.addPoint(
                float(x_c[i]), float(y_c[i]), 0.0, h_wall
            )
            airfoil_pts.append(tag)
        te_pt = airfoil_pts[0]

        airfoil_curves: List[int] = [gmsh.model.occ.addSpline(airfoil_pts)]
        airfoil_curves.append(gmsh.model.occ.addLine(airfoil_pts[-1], te_pt))
        airfoil_loop = gmsh.model.occ.addCurveLoop(airfoil_curves)

        # --- farfield rectangle (CCW) ---
        p_sw = gmsh.model.occ.addPoint(x_min, y_min, 0.0, h_far)
        p_se = gmsh.model.occ.addPoint(x_max, y_min, 0.0, h_far)
        p_ne = gmsh.model.occ.addPoint(x_max, y_max, 0.0, h_far)
        p_nw = gmsh.model.occ.addPoint(x_min, y_max, 0.0, h_far)
        l_s = gmsh.model.occ.addLine(p_sw, p_se)
        l_e = gmsh.model.occ.addLine(p_se, p_ne)
        l_n = gmsh.model.occ.addLine(p_ne, p_nw)
        l_w = gmsh.model.occ.addLine(p_nw, p_sw)
        outer_loop = gmsh.model.occ.addCurveLoop([l_s, l_e, l_n, l_w])

        surf = gmsh.model.occ.addPlaneSurface([outer_loop, airfoil_loop])
        gmsh.model.occ.synchronize()

        extruded = gmsh.model.occ.extrude(
            [(2, surf)],
            0,
            0,
            dz,
            numElements=[1],
            recombine=config.bl_recombine,
        )
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 1 if config.bl_recombine else 0)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 100)

        gmsh.model.mesh.setSize([(0, p) for p in airfoil_pts], h_wall)
        gmsh.model.mesh.setSize([(0, p) for p in (p_sw, p_se, p_ne, p_nw)], h_far)
        for curve in airfoil_curves:
            gmsh.model.mesh.setSize([(1, curve)], h_wall)
        _setup_boundary_layer(airfoil_curves, te_pt)

        gmsh.model.mesh.generate(3)

        # --- physical groups (surfaces for OpenFOAM patches) ---
        vol_tag = None
        front_tag = surf
        back_tag = None
        for dim, tag in extruded:
            if dim == 3:
                vol_tag = tag
            elif dim == 2 and tag != surf and back_tag is None:
                back_tag = tag

        airfoil_faces: List[int] = []
        inlet_faces: List[int] = []
        outlet_faces: List[int] = []
        top_faces: List[int] = []
        bottom_faces: List[int] = []
        for dim, tag in gmsh.model.getEntities(2):
            if tag in (front_tag, back_tag):
                continue
            kind = _classify_lateral_surface(
                dim, tag, chord, z_tol, x_min, x_max, y_min, y_max
            )
            if kind == "airfoil":
                airfoil_faces.append(tag)
            elif kind == "inlet":
                inlet_faces.append(tag)
            elif kind == "outlet":
                outlet_faces.append(tag)
            elif kind == "bottom":
                bottom_faces.append(tag)
            elif kind == "top":
                top_faces.append(tag)

        if airfoil_faces:
            gmsh.model.addPhysicalGroup(2, airfoil_faces, name="airfoil")
        if inlet_faces:
            gmsh.model.addPhysicalGroup(2, inlet_faces, name="inlet")
        if outlet_faces:
            gmsh.model.addPhysicalGroup(2, outlet_faces, name="outlet")
        if top_faces:
            gmsh.model.addPhysicalGroup(2, top_faces, name="top")
        if bottom_faces:
            gmsh.model.addPhysicalGroup(2, bottom_faces, name="bottom")

        cap_tags = [front_tag]
        if back_tag is not None:
            cap_tags.append(back_tag)
        if cap_tags:
            gmsh.model.addPhysicalGroup(2, cap_tags, name="frontAndBack")

        if vol_tag is not None:
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

        out_path = _mesh_file()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(out_path))

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"生成网格失败: {e}", file=sys.stderr)
        return False
    finally:
        if initialized:
            gmsh.finalize()

    return True


if __name__ == "__main__":
    from utils import default_coeffs

    coeffs = default_coeffs()
    if mesher(coeffs):
        print(f"✓ 网格文件已成功生成: {_mesh_file()}")
    else:
        print("✗ 网格文件生成失败", file=sys.stderr)
