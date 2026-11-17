"""Het Pakhuis, Haparandaweg 902-950 -> 4-colour ornament, from the game GLB (models/haparandaweg-902-950.glb)."""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
from scipy.spatial import ConvexHull
import orn, glbsrc, text

HEIGHT_MM = float(os.environ.get('ORN_HEIGHT_MM', 60))
SRC = os.path.join(ROOT, 'work/haparandaweg-902-950.raw.glb')
NAME, LETTERS, LIT = orn.variant_name('het-pakhuis')
if LIT:   # lit-from-inside: hollow shell, windows are single-wall white panes; trim shares the white filament
    PALETTE = [('brick', '42302A'), ('white', 'F4F2EC'), ('charcoal', '2E2E30')]
    C_TRIM = C_GLASS = 'white'
else:
    PALETTE = [('brick', '42302A'), ('cream', 'E6DCC6'), ('glass', '5D7380'), ('charcoal', '2E2E30')]
    C_TRIM, C_GLASS = 'cream', 'glass'
WALL_MM, PANE_MM = 1.6, 0.45

# ------------------------------------------------------------ load + square up the front
g = glbsrc.load(SRC, 320)
gl = glbsrc.components(g['glass'])
c = np.array([cc.bounds.mean(0) for cc in gl]); slope = np.polyfit(c[:, 0], c[:, 1], 1)[0]
g = glbsrc.load(SRC, 320 - math.degrees(math.atan(slope)))
src_h = g['wall'].bounds[1][2]
S = HEIGHT_MM / src_h                 # mm per metre
MM = 1.0 / S                          # metres per mm
TIN = 0.8 * MM                        # colour inlay depth
def _panel(cc):
    e = cc.extents
    return glbsrc.facing(cc) == 180 and e[1] < 0.06 and e[0] > 1.0 and e[2] > 1.0 and len(cc.faces) <= 4
stone_panels = [cc for cc in glbsrc.components(g['stone']) if _panel(cc)]
glass_c = glbsrc.components(g['glass']) + glbsrc.components(g['door']) + stone_panels
y_glass = np.median([cc.bounds[0][1] for cc in glass_c])
wall = g['wall']; n = wall.face_normals
front_sel = (n[:, 1] < -0.95)
ys = wall.triangles_center[front_sel][:, 1]; areas = wall.area_faces[front_sel]
y_front = float(ys[np.argmin(np.abs(ys - y_glass))]) if len(ys) else y_glass
y_front = min(y_front, y_glass)       # glass never in front of the wall face
print(f'scale 1:{1000 / S:.0f}  1mm = {MM:.3f} m  front y {y_front:.3f} glass y {y_glass:.3f}')

# ------------------------------------------------------------ massing from a heightfield of wall + roof
RES = 0.05
tris = np.r_[g['wall'].triangles, g['roof'].triangles]
tags = np.r_[np.zeros(len(g['wall'].faces), int), np.ones(len(g['roof'].faces), int)]
mn = tris.reshape(-1, 3).min(0); mx = tris.reshape(-1, 3).max(0)
x0, y0 = mn[0] - RES, mn[1] - RES
nx = int(np.ceil((mx[0] - x0) / RES)) + 2; ny = int(np.ceil((mx[1] - y0) / RES)) + 2
H, T = orn.rasterize_max(tris, x0, y0, RES, nx, ny, tags)
H[~np.isfinite(H)] = -1
# the front wall plane must be crisp: nothing in front of y_front
gy = y0 + np.arange(ny) * RES
j_in = int(np.argmax(gy >= y_front + 1e-6))
# Massing = the front profile (sawtooth gables, pitched roofs running back) extruded straight back to a
# flat rear facade. The game model's rear courtyard notch read as a mess; the rear is modelled from the
# user's photos below (2026-11-17).
gx = x0 + np.arange(nx) * RES
j_prof = j_in + int(round(1.0 / RES))
prof, tprof = H[j_prof].copy(), T[j_prof].copy()
ext = gy[(H > 0).any(1)].max()
Y_BACK = y_front + min(ext - y_front, 16.0)
j_b = int(np.searchsorted(gy, Y_BACK)) + 1
H[:j_b + 1, :] = prof; T[:j_b + 1, :] = tprof; H[j_b + 1:, :] = -1
body = orn.heightfield_solid(H, x0, y0, RES) ^ orn.box(-1e3, 1e3, y_front, Y_BACK, -1e3, 1e3)
H[:j_in, :] = -1
roof = orn.roof_skin(H, T, x0, y0, RES, 0.9 * MM, body)

# ------------------------------------------------------------ windows: merge panes per opening
RECESS = max(0.35 * MM, y_glass - y_front)
def bb2(cc): b = cc.bounds; return [b[0][0], b[0][2], b[1][0], b[1][2]]
boxes = [bb2(cc) for cc in glass_c]
parent = list(range(len(boxes)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
gap = 0.3
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i], boxes[j]
        if a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]:
            parent[find(i)] = find(j)
groups = {}
for i in range(len(boxes)): groups.setdefault(find(i), []).append(i)
cutters, glass_parts, outlines = [], [], []
# Uniform facade (user 2026-11-17): every bay of the upper floors repeats the left bay's windows
# (the source varies them around the old balconies), and each gable window is one opening with only
# its outer frame: no mullions, transoms or the band between the lower and pointed parts.
UP0, UP1, GAB = 7.0, 22.0, 22.2
hulls = [np.vstack([glass_c[i].vertices[:, [0, 2]] for i in ids]) for ids in groups.values()]
hulls = [p[ConvexHull(p).vertices] for p in hulls]
left_x = min(h[:, 0].min() for h in hulls)
bays = sorted({(round(h[:, 0].min(), 2), round(h[:, 0].max(), 2)) for h in hulls if h[:, 1].min() > 15 and h[:, 1].max() < UP1})
rows = sorted({(round(h[:, 1].min(), 2), round(h[:, 1].max(), 2)) for h in hulls
               if UP0 < h[:, 1].min() < UP1 and h[:, 0].min() < left_x + 0.3})
print('bays', bays, 'rows', rows)
from shapely.geometry import Polygon as _Poly, box as _sbox
from shapely.ops import unary_union
gable = {}
reg = []
for h in hulls:
    z0 = h[:, 1].min()
    if UP0 < z0 < UP1: continue                      # replaced by the regular grid below
    if z0 >= GAB:                                    # merge the gable's lower part, band and pointed head
        k = round(h[:, 0].mean()); gable.setdefault(k, []).append(_Poly(h)); continue
    reg.append(h)
for bx0, bx1 in bays:
    for z0, z1 in rows: reg.append(np.array([(bx0, z0), (bx1, z0), (bx1, z1), (bx0, z1)]))
GABLE_FRAME = max(0.3, 0.6 * MM)
gable_frames = []
for k, ps in gable.items():
    u = unary_union(ps); xb0, zb0, xb1, zb1 = u.bounds
    u = unary_union([u, _sbox(xb0, zb0, xb1, zb1).intersection(_Poly(np.vstack([p.exterior.coords for p in ps])).convex_hull)])
    u = u.convex_hull
    reg.append(np.array(u.exterior.coords)[:-1])
    gable_frames.append(u)
def _regular(cc):
    z0, z1 = cc.bounds[0][2], cc.bounds[1][2]
    return (8.0 < z0 < UP1 and z1 < UP1) or z0 >= GAB - 0.05
for hull in reg:
    outlines.append(hull)
    if LIT:
        cutters.append(orn.xz_prism(hull, y_front - 3, y_front + 3))
        glass_parts.append(orn.xz_prism(hull, y_front + RECESS, y_front + RECESS + PANE_MM * MM))
    else:
        cutters.append(orn.xz_prism(hull, y_front - 3, y_front + RECESS))
        glass_parts.append(orn.xz_prism(hull, y_front + RECESS, y_front + RECESS + TIN))
print('window openings', len(groups))
cut = orn.union(cutters)
if LIT:
    cav = orn.cavity_from_heightfield(H, x0, y0, RES, WALL_MM * MM)
    body = body - cav; roof = roof - cav
body = body - cut; roof = roof - cut

model = orn.Model(NAME, PALETTE)
model.set_priority(['brick', 'charcoal', 'white'] if LIT else ['brick', 'charcoal', 'glass', 'cream'])   # low -> high; roof charcoal sits on brick
model.add('brick', body)
model.add('charcoal', roof)
for gp in glass_parts: model.add(C_GLASS, gp)

# ------------------------------------------------------------ cream: floor bands, portal surrounds, copings
def fat_box(b, minx, minz, y0=None, y1=None):
    (ax, ay, az), (bx, by, bz) = b
    cx, cz = (ax + bx) / 2, (az + bz) / 2
    hx = max(bx - ax, minx) / 2; hz = max(bz - az, minz) / 2
    return orn.box(cx - hx, cx + hx, ay if y0 is None else y0, by if y1 is None else y1, cz - hz, cz + hz)
for cc in glbsrc.components(g['stone']):
    if _panel(cc) or _regular(cc): continue
    b = cc.bounds; f = glbsrc.facing(cc)
    if f == 180:
        proud = max(y_front - b[0][1], 0.2 * MM)
        cs = orn.facade_outline(cc, grow=0.14 * MM)
        if cs is not None: model.add(C_TRIM, orn.xz_prism(cs, y_front - proud, y_front + RECESS + TIN))
    elif b[0][1] < y_front + 0.3:
        proud = max(y_front - b[0][1], 0.2 * MM)
        model.add(C_TRIM, fat_box(b, 0.55 * MM, 0.35 * MM, y_front - proud, y_front + RECESS + TIN))
    # other stone belonged to the game model's rear massing, which is replaced below

SILL_PROUD = 0.16
for bx0, bx1 in bays:
    for z0, _ in rows[1:]:
        model.add(C_TRIM, orn.box(bx0 - 0.15, bx1 + 0.15, y_front - SILL_PROUD, y_front + RECESS + TIN, z0 - 0.37, z0 - 0.07))
for u in gable_frames:
    ring = u.difference(u.buffer(-GABLE_FRAME, join_style=2))
    model.add(C_TRIM, orn.xz_prism(orn.cs_shapely(ring), y_front - 0.2 * MM, y_front + RECESS + TIN))

# ------------------------------------------------------------ HET PAKHUIS, enlarged to print (orig 0.48 m letters ~ 1 mm)
font = text.load_font('/System/Library/Fonts/Supplemental/SuperClarendon.ttc', 7)
signs = sorted(glbsrc.components(g['sign']), key=lambda cc: cc.bounds[0][0])
letters = 'HETPAKHUIS'
ground_top = max(o[:, 1].max() for o in outlines if o[:, 1].min() < 4.5)
upper_bot = min(o[:, 1].min() for o in outlines if o[:, 1].min() > 6.5)
# letters sit between the ground-floor stone frames and the cream floor band, with clear brick above and
# below (user 2026-11-17: more separation)
_st = [cc.bounds for cc in glbsrc.components(g['stone'])]
frame_top = max([b[1][2] for b in _st if b[1][2] < 6.8] + [ground_top])
band_bot = min([b[0][2] for b in _st if 6.5 < b[0][2] < 8.0] + [upper_bot])
GAPM = 0.35 * MM
LH = min(2.4 * MM, (band_bot - frame_top) - 2 * GAPM)
zb = (frame_top + band_bot) / 2 - LH / 2
print(f'letters: gap {frame_top:.2f}-{band_bot:.2f} m, cap height {LH:.2f} m = {LH * S:.2f} mm')
for ch, cc in (zip(letters, signs) if LETTERS else []):
    xc = cc.bounds[:, 0].mean()
    cs, w = text.centred_letter(font, ch, xc, zb, LH)
    model.add(C_TRIM, orn.xz_prism(cs, y_front - 0.35 * MM, y_front + TIN))

# ------------------------------------------------------------ balconies: shallow relief with 45-degree undersides
P = 0.8 * MM
slabs = []   # uniform floors: balconies dropped so every upper-floor bay window reads the same (2026-11-17)
rails_glass = [cc.bounds for cc in glbsrc.components(g['glassRail'])]
for b in slabs:
    (ax, ay, az), (bx, by, bz) = b
    model.add(C_TRIM, orn.box(ax, bx, y_front - P, y_front + TIN, az, bz))
    yb = y_front + RECESS + TIN       # reach back to the glass / wall so the wedge is never loose
    wedge = [(y_front - P, az + 0.1 * MM), (y_front - P, az), (yb, az - (yb - y_front + P)), (yb, az + 0.1 * MM)]   # 45-degree underside
    model.add(C_TRIM, orn.yz_prism(wedge, ax, bx))
    rail_top = bz + 1.25
    is_glass = any(r[0][0] < bx and ax < r[1][0] and abs(r[0][2] - bz) < 0.2 for r in rails_glass)
    model.add(C_GLASS if is_glass else 'charcoal', orn.box(ax, bx, y_front - P, y_front + RECESS + TIN, bz, rail_top))

# louvres beside the left bay
for cc in glbsrc.components(g['louvre']):
    model.add('charcoal', fat_box(cc.bounds, 0.6 * MM, 0.35 * MM, y_front - 0.1 * MM, y_front + RECESS + TIN))

# ------------------------------------------------------------ rear facade, from the user's photos (2026-11-17)
# brick base (white doors, frosted shop glazing), five storeys of loggias in a taupe post-and-slab frame
# (charcoal filament) with railings, and a glazed triangle in each gable. Loggias are real recesses with
# 45-degree ceilings; railings are thin bands with open space behind them.
from shapely.geometry import Polygon as SPoly, box as sbox
yb = Y_BACK
on = prof > 0
xs_b = gx[on]; X0, X1 = xs_b.min(), xs_b.max() + RES
section = SPoly([(X0, 0)] + [(x + RES / 2, z) for x, z in zip(gx[on], prof[on])] + [(X1, 0)]).buffer(0)
bay_c = [(a + b) / 2 for a, b in bays]; BW = bays[1][0] - bays[0][0]
LZ0, LZ1, NL = 7.4, 21.9, 5
pitch = (LZ1 - LZ0) / NL
DR = 2.2 * MM; RT = 0.55 * MM; RAIL_H = 1.0; HALF = BW / 2 - 0.3
WALLm = WALL_MM * MM
rear_cut, rear_add = [], []
if LIT:   # a wall plate behind the loggias so they do not open into the cavity; 45-degree underside
    plate = orn.yz_prism([(yb - DR - WALLm, LZ0), (yb - DR - WALLm, LZ1 + 0.5), (yb, LZ1 + 0.5), (yb, LZ0 - DR - WALLm)], X0 + 0.01, X1 - 0.01)
    body = body + (plate ^ orn.box(-1e3, 1e3, -1e3, 1e3, -1e3, 1e3))
frame = sbox(X0, LZ0, X1, LZ1)
def rear_window(x0_, x1_, z0_, z1_, y_face, colour=None):
    poly = [(x0_, z0_), (x1_, z0_), (x1_, z1_), (x0_, z1_)]
    if colour:   # opaque door: shallow inlay
        rear_cut.append(orn.xz_prism(poly, y_face - TIN, y_face + 3)); model.add(colour, orn.xz_prism(poly, y_face - TIN, y_face)); return
    if LIT:
        rear_cut.append(orn.xz_prism(poly, y_face - 4, y_face + 3)); model.add(C_GLASS, orn.xz_prism(poly, y_face - RECESS - PANE_MM * MM, y_face - RECESS))
    else:
        rear_cut.append(orn.xz_prism(poly, y_face - RECESS, y_face + 3)); model.add(C_GLASS, orn.xz_prism(poly, y_face - RECESS - TIN, y_face - RECESS))
for c in bay_c:
    # base: ground-floor door + frosted window, first-floor pair of windows
    rear_window(c - 1.55, c - 0.6, 0.0, 2.4, yb, colour=C_TRIM)
    rear_window(c - 0.35, c + 1.55, 0.3, 2.9, yb)
    rear_window(c - 1.4, c - 0.2, 4.2, 6.6, yb); rear_window(c + 0.2, c + 1.4, 4.2, 6.6, yb)
    for i in range(NL):
        zb_ = LZ0 + i * pitch + 0.3; zt_ = LZ0 + (i + 1) * pitch
        frame = frame.difference(sbox(c - HALF, zb_, c + HALF, zt_))
        rear_cut.append(orn.yz_prism([(yb - DR, zb_), (yb + 1, zb_), (yb + 1, zt_), (yb, zt_), (yb - DR, zt_ - DR)], c - HALF, c + HALF))
        rear_add.append(('charcoal', orn.box(c - HALF, c + HALF, yb - RT, yb, zb_ - 0.01, zb_ + RAIL_H)))      # railing band
        for xa, xb_ in ((c - HALF + 0.25, c - 0.15), (c + 0.15, c + HALF - 0.25)):                          # glazed doors at the back of the loggia
            rear_window(xa, xb_, zb_ + 0.05, zt_ - DR - 0.15, yb - DR)
    # gable: one glazed triangle with a plain cream frame (matches the front)
    g = section.intersection(sbox(c - BW / 2 + 0.45, 22.5, c + BW / 2 - 0.45, 99)).buffer(-0.45, join_style=2)
    if not g.is_empty and g.area > 0.5:
        gp = np.array(g.exterior.coords)[:-1]
        if LIT:
            rear_cut.append(orn.xz_prism(gp, yb - 4, yb + 3)); model.add(C_GLASS, orn.xz_prism(gp, yb - RECESS - PANE_MM * MM, yb - RECESS))
        else:
            rear_cut.append(orn.xz_prism(gp, yb - RECESS, yb + 3)); model.add(C_GLASS, orn.xz_prism(gp, yb - RECESS - TIN, yb - RECESS))
        ring = g.difference(g.buffer(-GABLE_FRAME, join_style=2))
        rear_add.append((C_TRIM, orn.xz_prism(orn.cs_shapely(ring), yb - RECESS - TIN, yb + 0.2 * MM)))
rear_add.append(('charcoal', orn.xz_prism(orn.cs_shapely(frame), yb - TIN, yb + 0.15 * MM)))
rc = orn.union(rear_cut)
body = body - rc; roof = roof - rc
model.parts['brick'] = [body]; model.parts['charcoal'] = [roof] + model.parts['charcoal'][1:]
for col, part in rear_add: model.add(col, part)

parts, summary = orn.finish(model, S, os.path.join(ROOT, 'out', NAME), ring_key='charcoal')
print(json.dumps(summary, indent=1))
