"""Press-fit snap bottom for the lit (hollow, open-bottom) ornaments.

The cap is a 1.6 mm plate flush with the shell's footprint plus a plug that slides up inside the open bottom
(0.2 mm radial clearance, chamfered lead-in). A 0.3 mm bead round the plug clicks into a 0.35 mm groove cut into
the shell's inner wall, so the cap holds by a detent rather than friction and survives many open/close cycles.
A fingernail/coin notch under the plate's longest edge pries it off; no cable hole by default (a small battery LED sits inside).
All dimensions in mm (parts are already scaled and dropped to the bed).
"""
from __future__ import annotations
import numpy as np, trimesh
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box as sbox
from shapely.ops import unary_union

PLATE = 1.6          # base plate thickness
CLR = 0.2            # radial clearance plug <-> shell inner wall
PLUG_T = 1.4         # plug wall thickness
PLUG_H = 5.0         # plug height above the plate
LEAD = 0.6           # lead-in chamfer at the plug top
BEAD = 0.3           # bead protrusion beyond the plug (0.1 mm interference past the clearance)
BEAD_Z = 2.6         # bead centre, mm above the shell bottom
GROOVE = 0.35        # groove depth into the shell wall
NOTCH_W, NOTCH_D, NOTCH_H = 9.0, 1.6, 0.9
CABLE_D = 0.0        # no cable hole: a self-contained LED sits on the plate (set >0 for a fairy-light lead)
STEP = 0.1           # z step for the 45-degree tapers


def _slice(meshes, z):
    polys = []
    for m in meshes:
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None: continue
        p2, _ = sec.to_2D(to_2D=np.eye(4))
        polys += list(p2.polygons_full)
    return unary_union([p.buffer(0) for p in polys])


def _polys(g):
    if g.is_empty: return []
    return [g] if g.geom_type == 'Polygon' else [p for p in getattr(g, 'geoms', []) if p.geom_type == 'Polygon']


def _ext(poly, z0, z1):
    out = []
    for p in _polys(poly.buffer(0)):
        if p.area < 1e-3: continue
        m = trimesh.creation.extrude_polygon(p, z1 - z0); m.apply_translation([0, 0, z0]); out.append(m)
    return out


def _taper(base_poly, d_lo, d_hi, z0, z1):
    """Stack of thin slabs whose outward offset goes linearly from d_lo at z0 to d_hi at z1 (45-degree steps)."""
    n = max(1, int(round((z1 - z0) / STEP))); out = []
    for i in range(n):
        za, zb = z0 + (z1 - z0) * i / n, z0 + (z1 - z0) * (i + 1) / n
        d = d_lo + (d_hi - d_lo) * (i + 0.5) / n
        out += _ext(base_poly.buffer(d, join_style=2), za, zb)
    return out


def apply(parts: dict, manifold_union, to_manifold, to_trimesh):
    """parts: colour -> trimesh (mm, on bed). Returns (parts with the groove cut, cap trimesh, info)."""
    meshes = list(parts.values())
    zs = [0.25, 1.0, BEAD_Z, PLUG_H - 0.5, PLUG_H + 0.6]
    sl0 = _slice(meshes, 0.25)
    outer = unary_union([Polygon(p.exterior) for p in _polys(sl0) if p.area > 20])
    # free interior that stays open all the way up the plug height
    free = None
    for z in zs:
        s = _slice(meshes, z)
        holes = unary_union([Polygon(p.exterior) for p in _polys(s)]).difference(s)
        free = holes if free is None else free.intersection(holes)
    cav = [p for p in _polys(free.buffer(0)) if p.area > 30]
    if not cav: return parts, None, dict(error='no open cavity')
    cav = unary_union(cav)
    plug_out = cav.buffer(-CLR, join_style=2)
    plug_in = plug_out.buffer(-PLUG_T, join_style=2)
    plug_ring = plug_out.difference(plug_in)
    # cap (built with the plate at z<0 so the plug occupies shell coordinates 0..PLUG_H, then lifted to the bed)
    geo = _ext(outer, -PLATE, 0.0)
    geo += _ext(plug_out, -0.01, PLUG_H - LEAD)                       # solid footprint of the plug wall...
    hollow = _ext(plug_in, 0.0, PLUG_H + 1)                           # ...minus its inside
    geo += _taper(plug_ring.buffer(0), 0.0, -LEAD * 0.999, PLUG_H - LEAD, PLUG_H)  # lead-in chamfer (inward)
    zb0, zb1 = BEAD_Z - 0.6, BEAD_Z + 0.6
    bead = _taper(plug_out, 0.0, BEAD, zb0, BEAD_Z - 0.2) + _ext(plug_out.buffer(BEAD, join_style=2), BEAD_Z - 0.2, BEAD_Z + 0.2) \
         + _taper(plug_out, BEAD, 0.0, BEAD_Z + 0.2, zb1)
    geo += bead
    cap = manifold_union([to_manifold(m) for m in geo]) - manifold_union([to_manifold(m) for m in hollow])
    # pry notch under the longest outer edge
    ring = list(max(_polys(outer), key=lambda p: p.area).exterior.coords)
    a, b = max(zip(ring[:-1], ring[1:]), key=lambda e: np.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]))
    a, b = np.array(a), np.array(b); mid = (a + b) / 2; t = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([t[1], -t[0]])
    if outer.contains(Point(*(mid + nrm * 0.5))): nrm = -nrm
    c0 = mid - nrm * NOTCH_D
    notch = Polygon([c0 - t * NOTCH_W / 2, c0 + t * NOTCH_W / 2, c0 + t * NOTCH_W / 2 + nrm * 3, c0 - t * NOTCH_W / 2 + nrm * 3])
    cut = _ext(notch, -PLATE - 1, -PLATE + NOTCH_H)
    # cable hole: in the plate inside the plug, away from the walls
    inner_safe = plug_in.buffer(-CABLE_D)
    cable = None
    if CABLE_D > 0 and not inner_safe.is_empty:
        pt = inner_safe.representative_point(); cable = (pt.x, pt.y)
        cut.append(trimesh.creation.cylinder(radius=CABLE_D / 2, height=PLATE + 2, sections=32).apply_translation([pt.x, pt.y, -PLATE / 2]))
    cap = cap - manifold_union([to_manifold(m) for m in cut])
    cap = cap.translate([0, 0, PLATE])
    # groove in the shell: cavity grown by GROOVE with 45-degree flanks, cut from every colour
    groove = _taper(cav, 0.0, GROOVE, zb0 - 0.05, BEAD_Z - 0.2) + _ext(cav.buffer(GROOVE, join_style=2), BEAD_Z - 0.2, BEAD_Z + 0.2) \
           + _taper(cav, GROOVE, 0.0, BEAD_Z + 0.2, zb1 + 0.05)
    gm = manifold_union([to_manifold(m) for m in groove])
    new = {k: to_trimesh(to_manifold(p) - gm) for k, p in parts.items()}
    info = dict(cavity_mm2=round(cav.area, 1), plug_bbox=[round(v, 1) for v in plug_out.bounds],
                inner_free_bbox=[round(v, 1) for v in plug_in.bounds], cable_hole=cable is not None,
                plate_mm=PLATE, plug_mm=PLUG_H, clearance=CLR, bead=BEAD, groove=GROOVE)
    return new, to_trimesh(cap), info
