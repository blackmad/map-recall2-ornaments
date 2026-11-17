"""Ornament builder: printable multi-colour building ornaments.

Model space is metres, z up, front facade facing -y. Each ornament is a set of
Manifold solids per colour; `resolve()` makes them disjoint by priority, and
`finish()` scales to millimetres, adds the hanging ring, checks printability and
writes a Bambu/Orca-style 3MF plus per-colour STLs.
"""
from __future__ import annotations
import io, json, math, os, zipfile, datetime
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh, CrossSection, OpType, FillRule, JoinType

def cs_of(poly):
    if isinstance(poly, CrossSection): return poly
    p = np.asarray(poly, float)
    a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if a < 0: p = p[::-1]
    return CrossSection([p], FillRule.Positive)

# ---------------------------------------------------------------- primitives

def to_manifold(tm: trimesh.Trimesh) -> Manifold:
    tm = tm.copy(); tm.merge_vertices()
    m = Manifold(Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))
    if m.status().name != 'NoError':
        raise ValueError(f'not manifold: {m.status()}')
    return m

def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)

def box(x0, x1, y0, y1, z0, z1) -> Manifold:
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

def xz_prism(poly_xz, y0, y1) -> Manifold:
    """Polygon drawn on the facade (x right, z up), extruded from y0 to y1 (y into the building)."""
    y0, y1 = sorted((y0, y1))
    cs = cs_of(poly_xz)
    # extrude along +z then rotate so local z -> +y and local y -> +z
    m = cs.extrude(y1 - y0)            # (x, z_facade, y)
    m = m.rotate([90, 0, 0])            # rotate about x: (x, y, z) -> (x, -z, y)
    # after rotation: x stays, facade-z (local y) -> z, extrusion (local z) -> -y
    return m.mirror([0, 1, 0]).translate([0, y0, 0])

def yz_prism(poly_yz, x0, x1) -> Manifold:
    """Polygon in the side plane (y depth, z up), extruded along x from x0 to x1."""
    x0, x1 = sorted((x0, x1))
    cs = cs_of(poly_yz)
    m = cs.extrude(x1 - x0)             # (y, z, x)
    # map local (a, b, c) -> (c, a, b): rotate
    return m.rotate([90, 0, 90]).translate([x0, 0, 0])

def arch_poly(x0, x1, z0, zs, n=24):
    """Rectangle x0..x1 from z0 up to springing zs, topped with a semicircle."""
    r = (x1 - x0) / 2; cx = (x0 + x1) / 2
    pts = [(x0, z0), (x1, z0), (x1, zs)]
    for i in range(1, n):
        a = math.pi * i / n
        pts.append((cx + r * math.cos(a), zs + r * math.sin(a)))
    pts.append((x0, zs))
    return pts

def seg_arch_poly(x0, x1, z0, zs, rise, n=16):
    """Rectangle with a segmental (shallow) arch of given rise."""
    w = (x1 - x0) / 2; cx = (x0 + x1) / 2
    R = (w * w + rise * rise) / (2 * rise); cz = zs + rise - R
    a0 = math.acos(w / R)
    pts = [(x0, z0), (x1, z0)]
    for i in range(n + 1):
        a = a0 + (math.pi - 2 * a0) * i / n
        pts.append((cx + R * math.cos(a), cz + R * math.sin(a)))
    return pts

def union(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    if not ms: return Manifold()
    return Manifold.batch_boolean(ms, OpType.Add)

def torus(R, r, seg=48, rseg=20) -> Manifold:
    circ = CrossSection.circle(r, rseg).translate([R, 0])
    return circ.revolve(seg)

# ---------------------------------------------------------------- heightfield massing

def rasterize_max(tris: np.ndarray, x0, y0, res, nx, ny, tag=None):
    """Z-buffer (max) of triangles onto a vertex grid. tris: (n,3,3). Returns H (ny,nx) and tag ids."""
    H = np.full((ny, nx), -np.inf); T = np.full((ny, nx), -1, int)
    gx = x0 + np.arange(nx) * res; gy = y0 + np.arange(ny) * res
    for k, t in enumerate(tris):
        a, b, c = t
        mnx, mxx = t[:, 0].min(), t[:, 0].max(); mny, mxy = t[:, 1].min(), t[:, 1].max()
        i0 = max(0, int(np.floor((mnx - x0) / res))); i1 = min(nx - 1, int(np.ceil((mxx - x0) / res)))
        j0 = max(0, int(np.floor((mny - y0) / res))); j1 = min(ny - 1, int(np.ceil((mxy - y0) / res)))
        if i1 < i0 or j1 < j0: continue
        X, Y = np.meshgrid(gx[i0:i1 + 1], gy[j0:j1 + 1])
        v0 = b[:2] - a[:2]; v1 = c[:2] - a[:2]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12: continue  # vertical triangle: contributes via neighbours
        px = X - a[0]; py = Y - a[1]
        u = (px * v1[1] - v1[0] * py) / den; v = (v0[0] * py - px * v0[1]) / den
        eps = 1e-6
        inside = (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
        if not inside.any(): continue
        z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
        sub = H[j0:j1 + 1, i0:i1 + 1]; subT = T[j0:j1 + 1, i0:i1 + 1]
        upd = inside & (z > sub)
        sub[upd] = z[upd]
        if tag is not None: subT[upd] = tag[k]
    return H, T

def heightfield_solid(H: np.ndarray, x0, y0, res, floor=0.0) -> Manifold:
    """Closed solid under a vertex-grid heightfield. Cells with H<=floor are outside."""
    ny, nx = H.shape
    Hp = np.full((ny + 2, nx + 2), floor - 1.0); Hp[1:-1, 1:-1] = np.where(np.isfinite(H), np.maximum(H, floor - 1.0), floor - 1.0)
    ny, nx = Hp.shape; x0 -= res; y0 -= res
    gx = x0 + np.arange(nx) * res; gy = y0 + np.arange(ny) * res
    X, Y = np.meshgrid(gx, gy)
    top = np.c_[X.ravel(), Y.ravel(), Hp.ravel()]
    idx = np.arange(nx * ny).reshape(ny, nx)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel(); c = idx[1:, 1:].ravel(); d = idx[1:, :-1].ravel()
    ftop = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    # bottom: single quad at floor-1 using the 4 corners of the padded border (all at floor-1)
    corners = [idx[0, 0], idx[0, -1], idx[-1, -1], idx[-1, 0]]
    # border ring vertices
    ring = list(idx[0, :]) + list(idx[1:, -1]) + list(idx[-1, -2::-1]) + list(idx[-2:0:-1, 0])
    cen = len(top); verts = np.r_[top, [[gx.mean(), gy.mean(), floor - 1.0]]]
    fb = [[ring[(i + 1) % len(ring)], ring[i], cen] for i in range(len(ring))]
    faces = np.r_[ftop, np.array(fb)]
    tm = trimesh.Trimesh(verts, faces, process=False)
    if tm.volume < 0: tm.invert()
    m = to_manifold(tm)
    big = 1e4
    return m ^ box(-big, big, -big, big, floor, big)

# ---------------------------------------------------------------- colour resolution

class Model:
    def __init__(self, name, palette):
        """palette: list of (key, hex) in extruder order (max 4)."""
        self.name = name; self.palette = palette
        self.parts = {k: [] for k, _ in palette}
        self.priority = [k for k, _ in palette]   # later wins; override with set_priority

    def add(self, key, m):
        if m is not None and not m.is_empty(): self.parts[key].append(m)

    def set_priority(self, low_to_high):
        self.priority = list(low_to_high)

    def resolve(self):
        solids = {k: union(v) for k, v in self.parts.items()}
        out = {}; taken = Manifold()
        for k in reversed(self.priority):
            s = solids[k] - taken if not taken.is_empty() else solids[k]
            out[k] = s; taken = union([taken, solids[k]])
        return out

# ---------------------------------------------------------------- ring

def hanging_ring(body: Manifold, scale, ring_od_mm=11.0, tube_mm=2.4, post_mm=3.0, min_z=None, embed_z=None):
    """Ring in the facade plane (xz), centred over the centre of mass. Model units in metres."""
    tm = to_trimesh(body)
    com = tm.center_mass
    s = 1.0 / scale  # metres per mm
    r = tube_mm / 2 * s; R = (ring_od_mm / 2) * s - r
    # top surface height around COM
    zmax = top_height_near(tm, com[0], com[1], R + r)
    zc = zmax + R + r * 0.2
    if min_z is not None: zc = max(zc, min_z + R + r)   # stand the ring clear of e.g. a roof-terrace railing
    if embed_z is not None:     # 3/4 ring: the bottom quarter is sunk into the deck at embed_z, no post
        zc = embed_z + R * math.cos(math.radians(45))
        t = torus(R, r).rotate([90, 0, 0]).translate([com[0], com[1], zc])
        return t, t ^ box(com[0] - 1e3, com[0] + 1e3, com[1] - 1e3, com[1] + 1e3, -1e3, embed_z), com
    t = torus(R, r).rotate([90, 0, 0]).translate([com[0], com[1], zc])
    # post: from below the roof surface up into the ring
    lo = top_height_near(tm, com[0], com[1], post_mm / 2 * s, mode='min') - 2.0 * s
    post = box(com[0] - post_mm / 2 * s, com[0] + post_mm / 2 * s, com[1] - tube_mm / 2 * s * 1.0, com[1] + tube_mm / 2 * s * 1.0, lo, zc - R + r * 0.3)
    # flared boss so the post is not a weak neck on a sloped roof
    boss = box(com[0] - post_mm * s, com[0] + post_mm * s, com[1] - tube_mm * s, com[1] + tube_mm * s, lo, zmax + 0.4 * s)
    return union([t, post]), boss, com

def top_height_near(tm, x, y, rad, mode='max'):
    zs = []
    for dx in np.linspace(-rad, rad, 7):
        for dy in np.linspace(-rad, rad, 7):
            loc, _, _ = tm.ray.intersects_location([[x + dx, y + dy, 1e3]], [[0, 0, -1]])
            if len(loc): zs.append(loc[:, 2].max())
    if not zs: return tm.bounds[1][2]
    return max(zs) if mode == 'max' else min(zs)

# ---------------------------------------------------------------- export

def hex2rgb(h): return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def write_3mf(path, name, parts, palette):
    """parts: dict key -> trimesh (mm). One object made of one part per colour; extruder = palette order."""
    keys = [k for k, _ in palette if k in parts and len(parts[k].faces)]
    hexes = dict(palette)
    objs = []; comp_ids = []
    oid = 1
    xml_objs = []
    for k in keys:
        m = parts[k]
        v = '\n'.join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in m.vertices)
        t = '\n'.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
        xml_objs.append(f'<object id="{oid}" name="{k}" type="model"><mesh><vertices>\n{v}\n</vertices><triangles>\n{t}\n</triangles></mesh></object>')
        comp_ids.append((oid, k)); oid += 1
    parent = oid
    comps = ''.join(f'<component objectid="{i}"/>' for i, _ in comp_ids)
    xml_objs.append(f'<object id="{parent}" name="{name}" type="model"><components>{comps}</components></object>')
    # place on a 256x256 bed centre
    allv = np.vstack([parts[k].vertices for k in keys])
    mn, mx = allv.min(0), allv.max(0)
    tx, ty, tz = 128 - (mn[0] + mx[0]) / 2, 128 - (mn[1] + mx[1]) / 2, -mn[2]
    model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">
<metadata name="Application">map-recall2 ornament builder</metadata>
<metadata name="BambuStudio:3mfVersion">1</metadata>
<metadata name="Title">{name}</metadata>
<metadata name="CreationDate">{datetime.date.today().isoformat()}</metadata>
<resources>
{chr(10).join(xml_objs)}
</resources>
<build><item objectid="{parent}" transform="1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}" printable="1"/></build>
</model>'''
    parts_cfg = ''.join(
        f'<part id="{i}" subtype="normal_part"><metadata key="name" value="{k} ({hexes[k]})"/><metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/><metadata key="extruder" value="{keys.index(k) + 1}"/></part>'
        for i, k in comp_ids)
    cfg = f'''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="{parent}">
    <metadata key="name" value="{name}"/>
    <metadata key="extruder" value="1"/>
    {parts_cfg}
  </object>
  <plate><metadata key="plater_id" value="1"/><metadata key="locked" value="false"/>
    <model_instance><metadata key="object_id" value="{parent}"/><metadata key="instance_id" value="0"/><metadata key="identify_id" value="{parent}"/></model_instance>
  </plate>
</config>'''
    filament_colours = ';'.join('#' + hexes[k].upper() for k in keys)
    project = json.dumps({"filament_colour": ['#' + hexes[k].upper() for k in keys], "filament_type": ["PLA"] * len(keys)}, indent=1)
    ct = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/><Default Extension="config" ContentType="text/xml"/><Default Extension="png" ContentType="image/png"/></Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'''
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ct)
        z.writestr('_rels/.rels', rels)
        z.writestr('3D/3dmodel.model', model)
        z.writestr('Metadata/model_settings.config', cfg)
        z.writestr('Metadata/filament_colours.txt', filament_colours + '\n')
    return keys

# ---------------------------------------------------------------- checks

def printability_report(parts_mm: dict, min_feature=0.6, union_m=None):
    rep = {}
    allm = trimesh.util.concatenate(list(parts_mm.values()))
    for k, m in parts_mm.items():
        n = m.face_normals; a = m.area_faces
        z = m.triangles_center[:, 2]
        # downward faces not on the bed, steeper than 45 deg overhang
        over = (n[:, 2] < -0.71) & (z > 0.3)
        rep[k] = dict(watertight=bool(m.is_watertight), volume_mm3=round(float(m.volume), 1),
                      bodies=len(to_manifold(m).decompose()), overhang_mm2=round(float(a[over].sum()), 1))
    U = to_trimesh(union_m if union_m is not None else union([to_manifold(m) for m in parts_mm.values()]))
    n = U.face_normals; a = U.area_faces; zc = U.triangles_center[:, 2]
    over = (n[:, 2] < -0.72) & (zc > 0.3)
    zs = np.round(zc[over]).astype(int)
    rep['_exterior'] = dict(overhang_mm2=round(float(a[over].sum()), 2),
                            overhang_by_z={int(z): round(float(a[over][zs == z].sum()), 2) for z in np.unique(zs)},
                            bodies=len([c for c in to_manifold(U).decompose() if c.volume() > 0.01]))
    # pairwise overlap volume
    keys = list(parts_mm)
    ov = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            try:
                v = (to_manifold(parts_mm[keys[i]]) ^ to_manifold(parts_mm[keys[j]])).volume()
            except Exception as e:
                v = f'err {e}'
            ov[f'{keys[i]}&{keys[j]}'] = round(v, 3) if isinstance(v, float) else v
    return rep, ov

def thin_report(parts_mm: dict, pitch=0.1, min_feature=0.6):
    """Voxel thickness check on the union: fraction of solid voxels whose local thickness < min_feature."""
    from scipy import ndimage
    u = trimesh.util.concatenate(list(parts_mm.values()))
    vg = u.voxelized(pitch).fill()
    M = vg.matrix
    dt = ndimage.distance_transform_edt(M) * pitch
    # local thickness ~ 2*max inscribed radius over neighbourhood
    loc = ndimage.maximum_filter(dt, size=3) * 2
    thin = M & (loc < min_feature)
    return dict(voxels=int(M.sum()), thin_voxels=int(thin.sum()), thin_fraction=round(float(thin.sum()) / max(1, M.sum()), 5))

def finish(model: Model, scale_mm_per_m: float, out_dir: str, ring=True, ring_key=None, ring_min_z=None, ring_embed_z=None, ring_od_mm=11.0, ring_tube_mm=2.4):
    os.makedirs(out_dir, exist_ok=True)
    solids = model.resolve()
    if ring:
        body = union(list(solids.values()))
        ring_m, boss, com = hanging_ring(body, scale_mm_per_m, ring_od_mm=ring_od_mm, tube_mm=ring_tube_mm, min_z=ring_min_z, embed_z=ring_embed_z)
        rk = ring_key or model.priority[0]
        taken = union([v for k, v in solids.items() if k != rk])
        solids[rk] = union([solids[rk], (union([ring_m, boss]) - taken)])
    parts = {}
    dropped = {}
    man_mm = {}
    for k, s in solids.items():
        if s.is_empty(): continue
        s = s.scale([scale_mm_per_m] * 3)
        s = s.simplify(0.01)       # merge the heightfield's coplanar facets (<= 0.01 mm deviation)
        comps = s.decompose()
        keep = [c for c in comps if c.volume() >= 0.02]
        dropped[k] = len(comps) - len(keep)
        s = union(keep)
        if s.is_empty(): continue
        man_mm[k] = s
        tm = to_trimesh(s)
        parts[k] = tm
    # drop to bed
    zmin = min(p.bounds[0][2] for p in parts.values())
    for p in parts.values(): p.apply_translation([0, 0, -zmin])
    cap_info = None
    if model.name.endswith('-lit') and os.environ.get('ORN_NOCAP') != '1':
        import basecap
        parts, cap, cap_info = basecap.apply(parts, union, to_manifold, to_trimesh)
        if cap is not None:
            cdir = os.path.join(out_dir, 'base'); os.makedirs(cdir, exist_ok=True)
            ck = model.priority[0]; hx = dict(model.palette)[ck]
            cap.export(os.path.join(cdir, f'{model.name}-base__{ck}__{hx}.stl'))
            write_3mf(os.path.join(cdir, f'{model.name}-base.3mf'), model.name + '-base', {ck: cap}, model.palette)
            man_mm = {k: to_manifold(p) for k, p in parts.items()}
    hexes = dict(model.palette)
    for k, p in parts.items():
        p.export(os.path.join(out_dir, f'{model.name}__{k}__{hexes[k]}.stl'))
    keys = write_3mf(os.path.join(out_dir, f'{model.name}-4colour.3mf'), model.name, parts, model.palette)
    rep, ov = printability_report(parts, union_m=union(list(man_mm.values())))
    if os.environ.get('ORN_THIN') == '1':
        try: rep['_thin'] = thin_report(parts)
        except Exception as e: rep['_thin'] = str(e)
    allb = np.vstack([p.bounds for p in parts.values()])
    size = (allb.max(0) - allb.min(0)).round(1).tolist()
    summary = dict(dropped_slivers=dropped, name=model.name, size_mm=size, scale=f'1:{round(1000 / scale_mm_per_m)}', extruders={k: hexes[k] for k in keys}, parts=rep, overlaps_mm3=ov, base_cap=cap_info)
    json.dump(summary, open(os.path.join(out_dir, f'{model.name}-report.json'), 'w'), indent=1)
    return parts, summary

def facade_outline(tm, grow=0.0):
    """Union of the mesh's triangles projected on the facade (x,z), optionally grown by `grow` metres. Returns CrossSection."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for t in tm.triangles[:, :, [0, 2]]:
        p = Polygon(t)
        if p.area > 1e-8: polys.append(p)
    if not polys: return None
    u = unary_union(polys)
    if grow: u = u.buffer(grow, join_style=2)
    u = u.buffer(0)
    geoms = [u] if u.geom_type == 'Polygon' else list(u.geoms)
    contours = []
    for gm in geoms:
        if gm.area < 1e-6: continue
        contours.append(np.asarray(gm.exterior.coords)[:-1])
        contours += [np.asarray(i.coords)[:-1] for i in gm.interiors]
    if not contours: return None
    return CrossSection(contours, FillRule.EvenOdd)

def cavity_from_heightfield(H, x0, y0, res, wall, flat_slope=0.35):
    """Hollow interior for a lit ornament (open bottom).
    Ceiling follows the roof `wall` below it (footprint eroded by `wall`). Pitched ceilings print
    unsupported; every flat plateau of the ceiling is replaced by a 45-degree hipped pyramid so
    nothing needs bridging."""
    from scipy import ndimage
    r = max(1, int(np.ceil(wall / res)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; se = (xx * xx + yy * yy) <= r * r
    fp = H > 0
    Hmin = ndimage.grey_erosion(np.where(fp, H, -1.0), footprint=se)
    inside = ndimage.binary_erosion(fp, se)
    f0 = np.where(inside, Hmin - wall, -10.0)
    gyy, gxx = np.gradient(f0, res)
    flat = inside & (np.hypot(gxx, gyy) < flat_slope) & (f0 > 0)
    f = f0.copy()
    lab, n = ndimage.label(flat)
    for i in range(1, n + 1):
        reg = lab == i
        if reg.sum() < 4: continue
        d = ndimage.distance_transform_edt(reg) * res
        R = np.median(f0[reg])
        f[reg] = np.minimum(f0[reg], R - (d.max() - d[reg]))
    f[~inside] = -10.0
    f[f <= 0] = -10.0
    return heightfield_solid(f, x0, y0, res, floor=-wall)

def variant_name(base):
    letters = os.environ.get('ORN_LETTERS', '1') == '1'
    lit = os.environ.get('ORN_LIT', '0') == '1'
    return base + ('' if letters else '-noletters') + ('-lit' if lit else ''), letters, lit

def roof_skin(H, T, x0, y0, res, t, body):
    """Roof-coloured skin: the top `t` of cells tagged roof (T==1). The lowered field is grown sideways
    by two cells so its skirt lies outside the body's and no sliver of roof colour wraps the walls."""
    from scipy import ndimage
    Hl = np.where(T == 1, H - t, H)
    out = H <= 0
    grown = ndimage.maximum_filter(np.where(out, -1e9, Hl), size=5)
    Hl = np.where(out, np.maximum(grown, -1.0), Hl)
    return body - heightfield_solid(Hl, x0, y0, res)

def cs_shapely(geom):
    """shapely (Multi)Polygon with holes -> CrossSection (even-odd)."""
    geoms = [geom] if geom.geom_type == 'Polygon' else [g for g in getattr(geom, 'geoms', []) if g.geom_type == 'Polygon']
    contours = []
    for gm in geoms:
        if gm.area < 1e-9: continue
        contours.append(np.asarray(gm.exterior.coords)[:-1])
        contours += [np.asarray(i.coords)[:-1] for i in gm.interiors]
    return CrossSection(contours, FillRule.EvenOdd)
