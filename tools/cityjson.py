"""3DBAG CityJSON feature -> trimesh (local metres, z up, origin at footprint centroid on ground)."""
import json, numpy as np, trimesh
from trimesh.creation import triangulate_polygon
from shapely.geometry import Polygon

def _tri_face(ring_pts, holes):
    p = np.asarray(ring_pts, float)
    n = np.cross(p[1]-p[0], p[2]-p[0]); 
    # Newell normal
    n = np.zeros(3)
    for i in range(len(p)):
        a, b = p[i], p[(i+1) % len(p)]
        n += [(a[1]-b[1])*(a[2]+b[2]), (a[2]-b[2])*(a[0]+b[0]), (a[0]-b[0])*(a[1]+b[1])]
    n /= np.linalg.norm(n) + 1e-12
    u = np.cross(n, [0, 0, 1]) if abs(n[2]) < 0.9 else np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    o = p[0]
    to2 = lambda q: np.c_[(np.asarray(q)-o)@u, (np.asarray(q)-o)@v]
    poly = Polygon(to2(p), [to2(h) for h in holes])
    if not poly.is_valid: poly = poly.buffer(0)
    if poly.area < 1e-6: return None
    polys = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    out = []
    for pg in polys:
        v2, f = triangulate_polygon(pg, engine='earcut')
        v3 = o + v2[:, :1]*u + v2[:, 1:2]*v
        tm = trimesh.Trimesh(v3, f, process=False)
        if np.dot(tm.face_normals.mean(0), n) < 0: tm.invert()
        out.append(tm)
    return trimesh.util.concatenate(out)

def load(path, lod='2.2', origin=None):
    d = json.load(open(path))
    tr = d['metadata']['transform']; s = np.array(tr['scale']); t = np.array(tr['translate'])
    V = np.array(d['feature']['vertices']) * s + t
    meshes = []
    for oid, o in d['feature']['CityObjects'].items():
        for g in o.get('geometry', []):
            if str(g['lod']) != lod: continue
            for shell in g['boundaries']:
                for face in shell:
                    m = _tri_face(V[face[0]], [V[h] for h in face[1:]])
                    if m is not None: meshes.append(m)
    m = trimesh.util.concatenate(meshes); m.merge_vertices()
    ground = d['feature']['CityObjects'][[k for k in d['feature']['CityObjects'] if '-' not in k.split('.')[-1]][0]]['attributes']['b3_h_maaiveld']
    if origin is None: origin = np.r_[m.bounds.mean(0)[:2], ground]
    m.apply_translation(-np.asarray(origin))
    return m, origin
