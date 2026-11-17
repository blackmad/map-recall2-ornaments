"""Generic GLB -> printable ornament for buildings whose facades face any horizontal direction.

Roles per material:
  mass    : heightfield massing (walls)        roof : heightfield massing, coloured as a roof skin
  opening : windows/doors -> cut + coloured inlay (or white single-wall pane when lit)
  trim    : boxes thickened to printable widths, keyed into the wall, 45-degree undersides when they overhang
  rail    : balcony railings -> clustered into solid balustrades, squashed to <= PMAX protrusion
  drop    : ignored
"""
import os, math, numpy as np, trimesh
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import orn, glbsrc

def horiz_normal(c):
    n = c.face_normals; a = c.area_faces
    h = np.abs(n[:, 2]) < 0.7
    if not h.any() or a[h].sum() < 0.3 * a.sum(): return None
    # dominant horizontal axis (sign-free): weighted principal direction of normals
    v = n[h][:, :2] * np.sqrt(a[h])[:, None]
    w, U = np.linalg.eigh(v.T @ v)
    d = U[:, -1]
    return np.array([d[0], d[1], 0.0]) / np.linalg.norm(d)

class Frame:
    """Local facade frame: s along outward normal n, u along the facade (left->right seen from outside), w = z."""
    def __init__(self, n):
        self.n = np.asarray(n, float); self.n[2] = 0; self.n /= np.linalg.norm(self.n)
        self.u = np.array([-self.n[1], self.n[0], 0.0])
        self.theta = math.degrees(math.atan2(self.n[1], self.n[0])) + 90.0
    def uws(self, P):
        P = np.asarray(P)
        return np.c_[P @ self.u, P[:, 2], P @ self.n]
    def prism(self, poly_uw, s0, s1):
        """Polygon in (u, w), extruded along the outward normal from s0 to s1 (world)."""
        m = orn.xz_prism(poly_uw, -max(s0, s1), -min(s0, s1))  # local: x=u, z=w, -y = s
        return m.rotate([0, 0, self.theta])
    def box(self, u0, u1, w0, w1, s0, s1):
        return self.prism([(u0, w0), (u1, w0), (u1, w1), (u0, w1)], s0, s1)
    def side_prism(self, poly_sw, u0, u1):
        """Polygon in (s, w) (side section), extruded along u."""
        # local yz_prism: (y, z) polygon along x; local y = -s
        m = orn.yz_prism([(-s, w) for s, w in poly_sw], u0, u1)
        return m.rotate([0, 0, self.theta])

class Build:
    def __init__(self, name, glb, roles, palette, colour_of, height_mm, lit=False, res=0.03, front_bearing=180):
        self.g = glbsrc.load(glb, front_bearing)
        self.roles = roles; self.colour_of = colour_of; self.lit = lit
        top = max(m.bounds[1][2] for k, m in self.g.items() if roles.get(k) != 'drop')
        self.S = height_mm / top; self.MM = 1 / self.S; MM = self.MM
        self.TIN = 0.8 * MM; self.RECESS = 0.35 * MM; self.PANE = 0.45 * MM; self.WALL = 1.6 * MM
        self.PMAX = 1.0 * MM; self.MINW = 0.5 * MM; self.MINZ = 0.3 * MM
        self.res = res
        self.model = orn.Model(name, palette)
        self.name = name

    # ---------------------------------------------------------------- massing
    def massing(self):
        MM = self.MM
        mass = [k for k, r in self.roles.items() if r in ('mass', 'roof') and k in self.g]
        comps = []
        for k in mass:
            for c in glbsrc.components(self.g[k]): comps.append((k, c))
        base = [(k, c) for k, c in comps if c.bounds[0][2] < 0.3 or self.roles[k] == 'roof']
        foot = unary_union([Polygon(t[:, :2]).buffer(0) for k, c in base for t in c.triangles if Polygon(t[:, :2]).area > 1e-6])
        foot = foot.buffer(0.05).buffer(-0.05)
        polys = [foot] if foot.geom_type == 'Polygon' else list(foot.geoms)
        self.foot = unary_union([Polygon(p.exterior) for p in polys if p.area > 0.5])
        on_top = list(base); details = []
        rest = [(k, c) for k, c in comps if not any(c is cc for _, cc in base)]
        for k, c in rest:
            hull = Polygon(c.vertices[ConvexHull(c.vertices[:, :2]).vertices][:, :2]).buffer(0)
            frac = hull.intersection(self.foot).area / max(hull.area, 1e-9)
            (on_top if frac > 0.9 else details).append((k, c))
        self.mass_details = details
        tris = np.vstack([c.triangles for k, c in on_top])
        tags = np.concatenate([np.full(len(c.faces), 1 if self.roles[k] == 'roof' else 0) for k, c in on_top])
        res = self.res
        mn = tris.reshape(-1, 3).min(0); mx = tris.reshape(-1, 3).max(0)
        self.x0, self.y0 = mn[0] - res, mn[1] - res
        nx = int(np.ceil((mx[0] - self.x0) / res)) + 2; ny = int(np.ceil((mx[1] - self.y0) / res)) + 2
        H, T = orn.rasterize_max(tris, self.x0, self.y0, res, nx, ny, tags)
        H[~np.isfinite(H)] = -1
        self.H, self.T = H, T
        body = orn.heightfield_solid(H, self.x0, self.y0, res)
        roof_c = [k for k, r in self.roles.items() if r == 'roof']
        roof = orn.roof_skin(H, T, self.x0, self.y0, res, 0.9 * MM, body)
        if self.lit:
            cav = orn.cavity_from_heightfield(H, self.x0, self.y0, res, self.WALL)
            body = body - cav; roof = roof - cav
        self.body, self.roof = body, roof
        self.roof_colour = self.colour_of[roof_c[0]] if roof_c else None

    def height_at(self, x, y):
        i = int(round((x - self.x0) / self.res)); j = int(round((y - self.y0) / self.res))
        if 0 <= j < self.H.shape[0] and 0 <= i < self.H.shape[1]: return self.H[j, i]
        return -1

    def outward(self, c, n):
        """Orient horizontal axis n so it points out of the massing near component c."""
        ctr = c.bounds.mean(0)
        best = None
        for sgn in (1, -1):
            for d in (0.3, 0.6, 1.2, 2.0):
                p = ctr + sgn * n * d
                if self.height_at(p[0], p[1]) < ctr[2] - 0.05:
                    if best is None or d < best[1]: best = (sgn, d)
                    break
        return n * (best[0] if best else 1)

    def wall_s(self, fr, c):
        """Facade plane (s) behind component c along frame fr: march inward until inside the massing."""
        ctr = c.bounds.mean(0)
        s_c = ctr @ fr.n
        for d in np.arange(-0.6, 3.0, 0.02):
            p = ctr - fr.n * d
            if self.height_at(p[0], p[1]) > ctr[2]:
                return s_c - d
        return s_c

    # ---------------------------------------------------------------- openings
    def openings(self, mats):
        comps = []
        for k in mats:
            if k not in self.g: continue
            for c in glbsrc.components(self.g[k]):
                n = horiz_normal(c)
                if n is None: continue
                comps.append((k, c, self.outward(c, n)))
        # group: same direction, same plane, overlapping
        recs = []
        for k, c, n in comps:
            fr = Frame(n); P = fr.uws(c.vertices)
            recs.append(dict(k=k, c=c, n=n, fr=fr, P=P, box=[P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()], s=P[:, 2].max()))
        parent = list(range(len(recs)))
        def find(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        gap = 0.15
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if np.dot(a['n'], b['n']) < 0.97 or abs(a['c'].vertices.mean(0) @ a['n'] - b['c'].vertices.mean(0) @ a['n']) > 0.5: continue
                Pb = a['fr'].uws(b['c'].vertices)
                bb = [Pb[:, 0].min(), Pb[:, 1].min(), Pb[:, 0].max(), Pb[:, 1].max()]
                A = a['box']
                if A[0] - gap < bb[2] and bb[0] - gap < A[2] and A[1] - gap < bb[3] and bb[1] - gap < A[3]:
                    parent[find(i)] = find(j)
        groups = {}
        for i in range(len(recs)): groups.setdefault(find(i), []).append(i)
        cut, inl = [], []
        self.opening_boxes = []
        for ids in groups.values():
            r0 = recs[ids[0]]; fr = r0['fr']
            P = np.vstack([fr.uws(recs[i]['c'].vertices) for i in ids])
            hull = P[ConvexHull(P[:, :2]).vertices][:, :2]
            s_w = self.wall_s(fr, recs[ids[0]]['c'])
            s_g = s_w - self.RECESS
            colour = self.colour_of[recs[ids[0]]['k']]
            if self.lit:
                cut.append(fr.prism(hull, s_w - 3, s_w + 3))
                inl.append(('__pane__', fr.prism(hull, s_g - self.PANE, s_g)))
            else:
                cut.append(fr.prism(hull, s_g, s_w + 3))
                inl.append((colour, fr.prism(hull, s_g - self.TIN, s_g)))
            self.opening_boxes.append((fr, hull, s_w))
        c = orn.union(cut)
        self.body = self.body - c; self.roof = self.roof - c
        return inl

    # ---------------------------------------------------------------- trim
    def trim_piece(self, c, colour, keyed=True):
        """Return list of (colour, manifold) for one trim component."""
        out = []
        if c.is_watertight and len(c.faces) == 12:     # a box: keep it, widen thin sides to printable
            try:
                to, ext = trimesh.bounds.oriented_bounds(c)
                ext = np.maximum(ext, [self.MINW if e < self.MINW else e for e in ext])
                bx = orn.box(-ext[0] / 2, ext[0] / 2, -ext[1] / 2, ext[1] / 2, -ext[2] / 2, ext[2] / 2)
                M = np.linalg.inv(to)
                m = bx.transform(M[:3, :4])
                out.append((colour, m))
                hull = Polygon(trimesh.transform_points(np.array([[sx * ext[0] / 2, sy * ext[1] / 2, sz * ext[2] / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]), M)[:, :2]).convex_hull
                zb = c.bounds[0][2]
                if zb > 0.3: out += self.underside(hull, zb, colour)
                return out
            except Exception as e:
                print('box fallback', e)
        n = horiz_normal(c)
        if n is None:  # horizontal slab: OBB-ish axis-aligned in its own principal frame
            P2 = c.vertices[:, :2]
            hull = P2[ConvexHull(P2).vertices] if len(P2) >= 3 else None
            if hull is None: return out
            poly = Polygon(hull).buffer(0)
            mr = poly.minimum_rotated_rectangle
            # grow thin dimensions
            pts = np.asarray(mr.exterior.coords)[:4]
            e1 = pts[1] - pts[0]; e2 = pts[3] - pts[0]
            L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
            ctr = pts.mean(0)
            g1 = max(L1, self.MINW) / 2 / max(L1, 1e-9); g2 = max(L2, self.MINW) / 2 / max(L2, 1e-9)
            quad = [ctr - e1 * g1 - e2 * g2, ctr + e1 * g1 - e2 * g2, ctr + e1 * g1 + e2 * g2, ctr - e1 * g1 + e2 * g2]
            z0, z1 = c.bounds[0][2], c.bounds[1][2]
            if z1 - z0 < self.MINZ: zc = (z0 + z1) / 2; z0, z1 = zc - self.MINZ / 2, zc + self.MINZ / 2
            from manifold3d import CrossSection
            m = orn.cs_of(np.asarray(quad)).extrude(z1 - z0).translate([0, 0, z0])
            out.append((colour, m))
            out += self.underside(Polygon(quad), z0, colour)
            return out
        n = self.outward(c, n); fr = Frame(n)
        P = fr.uws(c.vertices)
        tri = fr.uws(c.triangles.reshape(-1, 3)).reshape(-1, 3, 3)[:, :, :2]
        pl = [Polygon(t) for t in tri]
        poly = unary_union([p for p in pl if p.area > 1e-9])
        if poly.is_empty or poly.area < 1e-6:   # edge-on: fall back to the hull of the projection
            poly = Polygon(P[ConvexHull(P[:, :2]).vertices][:, :2]).buffer(0) if len(P) > 3 else None
        if poly is None or poly.area < 1e-6: return out
        w_est = 2 * poly.area / max(poly.length, 1e-9)          # typical bar width (rings and strips)
        grow = max(0.0, (self.MINW - w_est) / 2)
        if grow > 0: poly = poly.buffer(grow, join_style='mitre')
        poly = poly.buffer(0)
        s_out = P[:, 2].max(); s_in = P[:, 2].min()
        s_w = self.wall_s(fr, c)
        if s_out - s_w < 0.2 * self.MM: s_out = s_w + 0.2 * self.MM     # at least a hair proud
        s_in = min(s_in, s_w - self.TIN) if keyed else s_in
        m = fr.prism(orn.cs_shapely(poly), s_in, s_out)
        out.append((colour, m))
        prot = s_out - s_w
        if prot > 0.6 * self.MM:   # 45-degree underside
            b = poly.bounds
            out.append((colour, fr.side_prism([(s_out, b[1] + 0.05 * self.MM), (s_out, b[1]), (s_w - 0.1 * self.MM, b[1] - (prot + 0.1 * self.MM)), (s_w - 0.1 * self.MM, b[1] + 0.05 * self.MM)], b[0], b[2])))
        return out

    def underside(self, quad, z0, colour):
        """45-degree support under the part of a horizontal slab that sticks out of the massing."""
        out = []
        outside = quad.difference(self.foot)
        if outside.area < (0.4 * self.MM) ** 2: return out
        # outward direction: from foot boundary nearest point to the outside centroid
        cen = outside.centroid
        q = self.foot.boundary.interpolate(self.foot.boundary.project(cen))
        v = np.array([cen.x - q.x, cen.y - q.y, 0.0]); dv = np.linalg.norm(v)
        if dv < 1e-6: return out
        fr = Frame(v / dv)
        xy = np.asarray(outside.convex_hull.exterior.coords)
        P = fr.uws(np.c_[xy, np.zeros(len(xy))])
        s_w = np.array([q.x, q.y, 0]) @ fr.n
        s_out = P[:, 2].max(); prot = s_out - s_w
        if prot < 0.6 * self.MM: return out
        u0, u1 = P[:, 0].min(), P[:, 0].max()
        out.append((colour, fr.side_prism([(s_out, z0 + 0.05 * self.MM), (s_out, z0), (s_w - 0.1 * self.MM, z0 - prot - 0.1 * self.MM), (s_w - 0.1 * self.MM, z0 + 0.05 * self.MM)], u0, u1)))
        return out

    # ---------------------------------------------------------------- balconies
    def balconies(self, rail_mats, slab_mats, rail_colour):
        """Cluster railings; each balcony becomes a solid balustrade + slab, squashed to PMAX protrusion."""
        rails = [c for k in rail_mats if k in self.g for c in glbsrc.components(self.g[k])]
        boxes = [c.bounds for c in rails]
        parent = list(range(len(rails)))
        def find(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        gap = 0.25
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if np.all(a[0] - gap < b[1]) and np.all(b[0] - gap < a[1]): parent[find(i)] = find(j)
        groups = {}
        for i in range(len(rails)): groups.setdefault(find(i), []).append(i)
        used_slabs = set(); out = []
        slab_comps = [(k, c) for k in slab_mats if k in self.g for c in glbsrc.components(self.g[k])]
        for ids in groups.values():
            pts = np.vstack([rails[i].vertices for i in ids])
            z0, z1 = pts[:, 2].min(), pts[:, 2].max()
            if z1 - z0 < 0.4: continue
            poly = Polygon(pts[ConvexHull(pts[:, :2]).vertices][:, :2]).buffer(0)
            outside = poly.difference(self.foot.buffer(-0.02))
            if outside.area < 0.05: continue
            # balcony direction from the nearest facade
            cen = outside.centroid; q = self.foot.boundary.interpolate(self.foot.boundary.project(cen))
            v = np.array([cen.x - q.x, cen.y - q.y, 0.0]); v /= np.linalg.norm(v)
            fr = Frame(v)
            s_w = np.array([q.x, q.y, 0.0]) @ fr.n
            P = fr.uws(pts)
            u0, u1 = P[:, 0].min(), P[:, 0].max(); s_out = P[:, 2].max()
            depth = s_out - s_w
            p = min(depth, self.PMAX)
            # slab: trim comps under the rails within this footprint
            zs = z0
            for idx, (k, c) in enumerate(slab_comps):
                b = c.bounds
                if b[1][2] <= z0 + 0.15 and b[1][2] >= z0 - 1.5 and Polygon(c.vertices[ConvexHull(c.vertices[:, :2]).vertices][:, :2]).buffer(0).intersects(outside):
                    used_slabs.add((k, id(c))); zs = min(zs, b[0][2])
            slab_t = min(max(z0 - zs, 0.25), 0.4)
            zs = z0 - slab_t
            out.append((self.colour_of[slab_mats[0]], fr.box(u0, u1, zs, z0, s_w - self.TIN, s_w + p)))
            out.append((rail_colour, fr.box(u0, u1, z0, z1, s_w - (self.TIN if not self.lit else self.TIN), s_w + p)))
            out.append((self.colour_of[slab_mats[0]], fr.side_prism([(s_w + p, zs + 0.05 * self.MM), (s_w + p, zs), (s_w - 0.1 * self.MM, zs - p - 0.1 * self.MM), (s_w - 0.1 * self.MM, zs + 0.05 * self.MM)], u0, u1)))
            out.append(('__balcony_zone__', (fr, u0, u1, zs - p, z1, s_w)))
        self.used_slabs = used_slabs
        return out
