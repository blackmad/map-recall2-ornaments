"""Load a decoded game GLB into a z-up facade frame and group connected components by material."""
import numpy as np, trimesh, collections
def load(path, front_bearing):
    s = trimesh.load(path)
    R = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]], float)
    th = np.radians(front_bearing - 180)          # rotate CCW by th: bearing b -> b - th
    c, s_ = np.cos(th), np.sin(th)
    Rz = np.array([[c,-s_,0,0],[s_,c,0,0],[0,0,1,0],[0,0,0,1]])
    out = {}
    for g in s.dump():
        g = g.copy(); g.apply_transform(Rz @ R)
        out[g.visual.material.name] = g
    return out
def components(g):
    g = g.copy(); g.merge_vertices(merge_tex=True, merge_norm=True)
    return g.split(only_watertight=False)
def facing(c):
    n = c.face_normals; a = c.area_faces
    best = collections.Counter()
    for nn, aa in zip(n, a):
        if abs(nn[2]) > 0.7: k = 'up' if nn[2] > 0 else 'down'
        else: k = int(round(np.degrees(np.arctan2(nn[0], nn[1])) / 10) * 10) % 360
        best[k] += aa
    return best.most_common(1)[0][0]
