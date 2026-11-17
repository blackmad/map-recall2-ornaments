import sys, os, numpy as np
sys.path.insert(0, 'tools')
import orn, glbsrc, glbbuild
ROLES = dict(brick='mass', slate='roof', glass='opening', red='opening', white='trim', stone='trim', concrete='trim', dark='drop', bronze='rail', green='drop')
COLOUR = dict(brick='brick', slate='slate', glass='dark', red='dark', white='white', stone='white', concrete='white', dark='dark', bronze='dark', green='dark')
PALETTE = [('brick', '6B4034'), ('white', 'EEEAE0'), ('dark', '27303A'), ('slate', '4E5459')]
B = glbbuild.Build('x', 'work/da-costakade-13.raw.glb', ROLES, PALETTE, COLOUR, 60, res=0.05)
B.massing()
print('materials', {k: len(v.faces) for k, v in B.g.items()})
f = B.foot; print('foot', f.geom_type, f.area, np.round(f.bounds, 2))
c = np.array(f.exterior.coords)
for a, b in zip(c[:-1], c[1:]):
    L = np.linalg.norm(b - a)
    if L > 0.8: d = (b - a) / L; print('edge', np.round(a, 2), np.round(b, 2), 'len', round(L, 2), 'outward n', np.round([d[1], -d[0]], 2))
print('top z', B.H.max())
B.openings(['glass', 'red'])
rows = []
for fr, hull, s_w in B.opening_boxes:
    rows.append((tuple(np.round(fr.n[:2], 2)), round(s_w, 2), round(hull[:, 0].min(), 2), round(hull[:, 0].max(), 2), round(hull[:, 1].min(), 2), round(hull[:, 1].max(), 2), len(hull)))
for r in sorted(rows): print(r)
for k in ('bronze', 'white', 'stone', 'concrete', 'green', 'dark'):
    if k in B.g:
        cs = glbsrc.components(B.g[k]); print(k, len(cs), 'comps; z range', np.round(B.g[k].bounds[:, 2], 2))
fr2 = [f for f, h, s in B.opening_boxes if abs(f.n[0] + 0.37) < 0.05][0]
c = np.array(B.foot.exterior.coords)
for p in c[:-1]: print('pt', np.round(p, 2), 'E2 u', round(p @ fr2.u[:2], 2), 's', round(p @ fr2.n[:2], 2))
mid = np.array([(-3 + 1.55) / 2, (3.14 + 4.93) / 2]) - fr2.n[:2] * 0.4
print('E2 eave z at mid', B.height_at(*mid), [round(B.height_at(*(mid - fr2.n[:2] * d)), 2) for d in (0, 1, 2, 3, 4, 6)])
H = B.H
vals, cnt = np.unique(np.round(H[H > 12], 1), return_counts=True)
print('top H histogram', [(v, c) for v, c in zip(vals, cnt) if c > 30])
for nm, p in [('E2 corner a', (-3.0, 3.14)), ('E2 corner b', (1.55, 4.93)), ('E2 mid', (-0.72, 4.03))]:
    q = np.array(p) - fr2.n[:2] * 0.3
    print(nm, B.height_at(*q))
