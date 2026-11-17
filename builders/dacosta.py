"""Da Costakade 13 (corner block, 1904) -> ornament.

Massing (footprint, mansard, gable) comes from the game GLB; every facade detail is procedural, from the
user's photos and the makelaar floorplans (2026-11-17):
  * gable front (E2): two bays of French doors with segmental heads; balconettes with open black railings on
    floors 1-3, carried by stepped brick corbels; green-grey glazed-brick arches over every window, string
    courses at sill and springing level, quoin strips at the gable corners.
  * no. 13 front (E1): one wide segmental window + a pair of narrow round-headed windows per floor, round-arched
    entrance, mansard dormer.
  * rear (E4) + side (E3): floors 1-2 have a 4.37 x 0.95 m balcony on the party-wall half of the rear and a
    1.57 m side balcony on E3 by the rear corner (on rust posts with white braces); floor 3 has a 0.97 m balcony
    along the whole rear that wraps 2.98 m down the side. Other rear windows get Juliet railings.
  * roof: the flat top is an inset roof terrace behind a parapet with a black railing.
  * blue-grey plinth all round (slate filament).
Palette: brick, white, dark (glass + iron), slate (green-grey accents, plinth, roof).
The previous GLB-trim version is builders/dacosta_glb_v1.py.
"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np, trimesh
from shapely.geometry import Polygon as SP, Point, box as sbox
from shapely.ops import unary_union
from scipy import ndimage
from skimage import measure
import orn, glbbuild
from glbbuild import Frame

NAME, LETTERS, LIT = orn.variant_name('da-costakade-13')
PALETTE = [('brick', '7A4434'), ('white', 'EEEAE0'), ('dark', '23292F'), ('slate', '4E5A53')]
ROLES = dict(brick='mass', slate='roof', glass='opening', red='opening', white='drop', stone='drop', concrete='drop',
             dark='drop', bronze='drop', green='drop')
COLOUR = dict(brick='brick', slate='slate', glass='dark', red='dark')
B = glbbuild.Build(NAME, os.path.join(ROOT, 'work/da-costakade-13.raw.glb'), ROLES, PALETTE, COLOUR,
                   float(os.environ.get('ORN_HEIGHT_MM', 60)), lit=False, res=0.04)
S, MM = B.S, B.MM
TIN, RECESS, PANE, WALL = B.TIN, B.RECESS, B.PANE, B.WALL
FW = max(0.15, 0.5 * MM)            # white frame bar
RT = 0.55 * MM                      # railing bar
KICK, TOPR = 0.4 * MM, 0.4 * MM
BAND = max(0.15, 0.5 * MM)          # green string course height
print(f'scale 1:{1000 / S:.0f}, 1 mm = {MM:.3f} m')
B.massing()

# ------------------------------------------------------------------ roof terrace (inset into the flat top)
H, T, res, x0, y0 = B.H, B.T, B.res, B.x0, B.y0
ZFLAT = 16.4
H[H > ZFLAT] = ZFLAT        # drop the game model's gable block behind the gable face; the new gable is added below
flat = np.abs(H - ZFLAT) < 0.06
lab, nlab = ndimage.label(flat); flat = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
PARA = max(0.36, 1.2 * MM)
r = int(round(PARA / res))
yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
terr = ndimage.binary_erosion(flat, structure=(xx * xx + yy * yy) <= r * r)
TDEPTH = 0.95
H[terr] = ZFLAT - TDEPTH
cs = max(measure.find_contours(flat.astype(float), 0.5), key=len)
flat_poly = SP([(x0 + c[1] * res, y0 + c[0] * res) for c in cs]).buffer(0).simplify(0.08)
body = orn.heightfield_solid(H, x0, y0, res)
# crisp walls: clip the raster to the real footprint and fill straight walls up to the mansard foot
foot_cs = orn.cs_shapely(B.foot)
body = (body ^ orn.to_manifold(trimesh.creation.extrude_polygon(B.foot, 40)).translate([0, 0, -1])) + \
       orn.to_manifold(trimesh.creation.extrude_polygon(B.foot, 13.9))
roof = orn.roof_skin(H, T, x0, y0, res, 0.9 * MM, body)
GABLE = []   # filled below, once the facade frames exist
if LIT:
    cav = orn.cavity_from_heightfield(H, x0, y0, res, WALL)
    body = body - cav; roof = roof - cav

# ------------------------------------------------------------------ facades
foot = B.foot
def outward_frame(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([d[1], -d[0], 0.0]); mid = (a + b) / 2
    if foot.contains(Point(*(mid + nrm[:2] * 0.3))): nrm = -nrm
    fr = Frame(nrm)
    ua, ub = sorted([np.r_[a, 0] @ fr.u, np.r_[b, 0] @ fr.u])
    return dict(fr=fr, s=float(np.r_[a, 0] @ fr.n), u0=ua, u1=ub)
FAC = dict(E1=outward_frame((-7.72, 0.09), (-3.46, 2.91)), E2=outward_frame((-3.0, 3.14), (1.55, 4.93)),
           E3=outward_frame((1.56, 4.93), (7.13, -0.2)), E4=outward_frame((7.13, -0.2), (0.5, -6.65)),
           E5=outward_frame((-0.21, -6.08), (-7.72, 0.09)))
for k, f in FAC.items(): print(k, 'n', np.round(f['fr'].n[:2], 2), 's', round(f['s'], 2), 'u', round(f['u0'], 2), round(f['u1'], 2))

# steep pointed gable on E2 (photo: ~58 degrees, finial, stone copings), replacing the game model's low gable
fg = FAC['E2']; frg = fg['fr']
ZE, UC, HW = 14.4, (fg['u0'] + fg['u1']) / 2, (fg['u1'] - fg['u0']) / 2
APEX = ZE + HW * math.tan(math.radians(58))
tri = [(fg['u0'], ZE - 0.5), (fg['u1'], ZE - 0.5), (fg['u1'], ZE), (UC, APEX), (fg['u0'], ZE)]
def lowered(dz): return [(fg['u0'], ZE - 0.5), (fg['u1'], ZE - 0.5), (fg['u1'], ZE - dz), (UC, APEX - dz), (fg['u0'], ZE - dz)]
GD = max(0.6, 2.0 * MM)     # screen gable: nothing rises behind it (user 2026-11-17)
gab = frg.prism(tri, fg['s'] - GD, fg['s'])
body = body + gab
dz_skin = 0.9 * MM / math.cos(math.radians(58))
skin = frg.prism(tri, fg['s'] - GD, fg['s'] - 0.2) - frg.prism(lowered(dz_skin), fg['s'] - GD - 1, fg['s'])
cope = frg.prism(tri, fg['s'] - 0.3, fg['s'] + max(0.12, 0.4 * MM)) - frg.prism(lowered(0.26), fg['s'] - 1, fg['s'] + 1)
fin = frg.box(UC - 0.16, UC + 0.16, APEX - 0.4, APEX + 0.75, fg['s'] - 0.35, fg['s'] + 0.05)
roof = roof - frg.prism([(fg['u0'], ZE - 1.6), (fg['u1'], ZE - 1.6), (fg['u1'], ZE), (UC, APEX), (fg['u0'], ZE)], fg['s'] - GD - 0.01, fg['s'] + 0.3)
body = body - skin - cope - fin
roof = roof + (skin - cope - fin)
GABLE.append(('white', cope + fin))     # stone copings + finial, embedded in the gable
if LIT and GD > 1.5:
    body = body - frg.prism([(fg['u0'] + WALL, ZE - 1), (fg['u1'] - WALL, ZE - 1), (UC, APEX - WALL * 2)], fg['s'] - GD + WALL, fg['s'] - WALL)
print(f'gable apex {APEX:.2f} m')

# openings from the GLB, re-expressed in each facade's frame
body0, roof0 = B.body, B.roof
B.openings(['glass', 'red']); B.body, B.roof = body0, roof0
OPEN = {k: [] for k in FAC}
for fro, hull, s_w in B.opening_boxes:
    best = max(FAC, key=lambda k: fro.n @ FAC[k]['fr'].n)
    f = FAC[best]
    if fro.n @ f['fr'].n < 0.95: continue
    us = [((u * fro.u + s_w * fro.n) @ f['fr'].u) for u in (hull[:, 0].min(), hull[:, 0].max())]
    OPEN[best].append(dict(u0=min(us), u1=max(us), w0=hull[:, 1].min(), w1=hull[:, 1].max(), s=s_w))
for k in OPEN: OPEN[k].sort(key=lambda o: (o['w0'], o['u0']))
print({k: len(v) for k, v in OPEN.items()})
# rear (E4): the stair column's windows sit half a level low in the game model; the user wants them on the floor rows
ROWS4 = [(4.7, 6.65), (8.1, 10.05), (11.5, 13.45)]
e4 = []
for o in OPEN['E4']:
    if o['w0'] < 3: continue                     # ground floor rebuilt below
    r0, r1 = min(ROWS4, key=lambda r: abs(r[0] - o['w0']))
    e4.append(dict(o, w0=r0, w1=r1))
g = [o for o in OPEN['E4'] if o['w0'] < 3]
if g:                                            # one wide glazed garden door across the two left columns
    e4.append(dict(g[0], u0=min(o['u0'] for o in g), u1=max(o['u1'] for o in g), w0=0.15, w1=2.95, garden=True))
OPEN['E4'] = sorted({(round(o['u0'], 2), round(o['w0'], 2)): o for o in e4}.values(), key=lambda o: (o['w0'], o['u0']))


m = orn.Model(NAME, PALETTE)
m.set_priority(['brick', 'slate', 'dark', 'white'])
cuts, adds, accents, accent_holes = [], [], [], []

def window(fk, o, kind, mullion=False, transom=True, arch_band=True, door=False):
    fr = FAC[fk]['fr']; s_w = o['s']
    u0, u1, w0, w1 = o['u0'], o['u1'], o['w0'], o['w1']
    wid = u1 - u0
    if kind == 'seg':
        rise = min(0.2 * wid, 0.35); spring = w1 - rise
        poly = orn.seg_arch_poly(u0, u1, w0, spring, rise)
    elif kind == 'round':
        spring = w1 - wid / 2; poly = orn.arch_poly(u0, u1, w0, spring)
    else:
        spring = w1; poly = [(u0, w0), (u1, w0), (u1, w1), (u0, w1)]
    P = SP(poly).buffer(0)
    if LIT and not door:
        cuts.append(fr.prism(poly, s_w - 1.2, s_w + 3)); adds.append(('white', fr.prism(poly, s_w - RECESS - PANE, s_w - RECESS)))
    else:
        cuts.append(fr.prism(poly, s_w - RECESS, s_w + 3)); adds.append(('dark', fr.prism(poly, s_w - RECESS - TIN, s_w - RECESS)))
    if not door:
        ring = P.difference(P.buffer(-FW, join_style=2))
        adds.append(('white', fr.prism(orn.cs_shapely(ring), s_w - RECESS - TIN, s_w - 0.05 * MM)))
        bars = []
        if transom and w1 - w0 > 1.6:
            zt = spring if kind != 'rect' else w1 - 0.55
            bars.append(sbox(u0, zt - FW, u1, zt))
        if mullion and wid > 0.9:
            uc = (u0 + u1) / 2; bars.append(sbox(uc - FW / 2, w0, uc + FW / 2, spring))
        for b in bars: adds.append(('white', fr.prism(orn.cs_shapely(b), s_w - RECESS - TIN, s_w - RECESS + 0.1 * MM)))
        adds.append(('white', fr.box(u0 - 0.06, u1 + 0.06, w0 - 0.12, w0, s_w - TIN, s_w + 0.25 * MM)))   # sill
    accent_holes.append((fk, P.buffer(0.04)))
    if arch_band and kind != 'rect':
        band = P.buffer(0.26, join_style=2).difference(P.buffer(0.02)).intersection(sbox(u0 - 1, spring - 0.05, u1 + 1, 99))
        accents.append((fk, band, s_w))
    return spring

def open_rail(fr, u0, u1, s_front, zf, h=1.0, side_ends=(), s_wall=None):
    """Open railing along u at s_front (kick plate, hand rail, posts); side returns back to s_wall."""
    out = [fr.box(u0, u1, zf, zf + KICK, s_front - RT, s_front), fr.box(u0, u1, zf + h - TOPR, zf + h, s_front - RT, s_front)]
    n_post = max(2, int(round((u1 - u0) / 0.45)) + 1)
    for i in range(n_post):
        up = u0 + (u1 - u0 - RT) * i / (n_post - 1)
        out.append(fr.box(up, up + RT, zf, zf + h, s_front - RT, s_front))
    for ue in side_ends:
        ua = ue if ue < (u0 + u1) / 2 else ue - RT
        out.append(fr.box(ua, ua + RT, zf, zf + KICK, s_wall - 0.02, s_front))
        out.append(fr.box(ua, ua + RT, zf + h - TOPR, zf + h, s_wall - 0.02, s_front))
        nb = max(2, int(round((s_front - s_wall) / 0.45)) + 1)
        for i in range(nb):
            sp = s_wall + (s_front - s_wall - RT) * i / (nb - 1)
            out.append(fr.box(ua, ua + RT, zf, zf + h, sp, sp + RT))
    return [('dark', o) for o in out]

SLAB = max(0.16, 0.5 * MM)
# ------------------------------------------------------------------ E2: gable front
fr2 = FAC['E2']['fr']
P_BAL = max(0.55, 1.9 * MM)
for o in OPEN['E2']:
    if o['w0'] > 14:      # gable window, moved onto the new gable face and sized from the photo
        o = dict(o, s=fg['s'], u0=UC - 0.78, u1=UC + 0.78, w0=ZE + 0.45, w1=ZE + 2.25)
        window('E2', o, 'round', mullion=True); continue
    window('E2', o, 'seg', mullion=True)
    if 3.5 < o['w0'] < 12:                               # balconette floors 1-3
        s_w = o['s']; ub0, ub1 = o['u0'] - 0.28, o['u1'] + 0.28; zf = o['w0']
        adds.append(('white', fr2.box(ub0, ub1, zf - SLAB, zf, s_w - TIN, s_w + P_BAL)))
        adds += open_rail(fr2, ub0, ub1, s_w + P_BAL, zf, side_ends=[ub0, ub1], s_wall=s_w)
        zt = zf - SLAB; P = P_BAL
        prof = [(s_w - TIN, zt), (s_w + P, zt), (s_w + P, zt - 0.25), (s_w + 2 * P / 3, zt - 0.25), (s_w + 2 * P / 3, zt - 0.5),
                (s_w + P / 3, zt - 0.5), (s_w + P / 3, zt - 0.75), (s_w - TIN, zt - 0.75)]
        for ca, cb in ((ub0, ub0 + 0.36), (ub1 - 0.36, ub1)):
            adds.append(('brick', fr2.side_prism(prof, ca, cb)))

# ------------------------------------------------------------------ E1: no. 13 street front
for o in OPEN['E1']:
    wid = o['u1'] - o['u0']
    if o['w0'] > 14: window('E1', o, 'rect', transom=False, arch_band=False)       # mansard dormer
    elif o['w0'] < 0.5 and wid > 1.2: window('E1', o, 'round', door=True)         # arched entrance
    elif wid < 0.8: window('E1', o, 'round')
    else: window('E1', o, 'seg')

# ------------------------------------------------------------------ E4 rear, E3 side: balconies
f3, f4 = FAC['E3'], FAC['E4']; fr3, fr4 = f3['fr'], f4['fr']
C = np.array([7.13, -0.2, 0.0]); O4 = np.array([0.5, -6.65, 0.0])     # E3/E4 corner, E4 party-wall end
uC4, uO4 = C @ fr4.u, O4 @ fr4.u; d4 = np.sign(uO4 - uC4)
uC3 = C @ fr3.u; uF3 = np.array([1.56, 4.93, 0.0]) @ fr3.u; d3 = np.sign(uF3 - uC3)
s3, s4 = f3['s'], f4['s']
BD, WD = 0.97, 1.05
def along(u_start, d, a, b): return sorted([u_start + d * a, u_start + d * b])
from collections import Counter     # floor sills = the most common rear sill heights (stair windows sit half a level lower)
cnt = Counter(round(o['w0'], 1) for o in OPEN['E4'] if o['w0'] > 3)
lv = sorted(v for v, c in cnt.items() if c >= 2)
print('rear levels', [round(v, 2) for v in lv])
balc_zones = []
def slab_and_rail(fr, s_w, ua, ub, zf, depth, ends, brackets=True):
    adds.append(('white', fr.box(ua, ub, zf - SLAB, zf, s_w - TIN, s_w + depth)))
    adds.extend(open_rail(fr, ua, ub, s_w + depth, zf, side_ends=ends, s_wall=s_w))
    if brackets:     # white 45-degree brackets every ~2 m (the photographed white braces)
        nbk = max(2, int(math.ceil((ub - ua) / 2.0)) + 1)
        zt = zf - SLAB
        for i in range(nbk):
            uk = ua + (ub - ua - 0.18) * i / (nbk - 1)
            adds.append(('white', fr.side_prism([(s_w - TIN, zt), (s_w + depth, zt), (s_w - TIN, zt - depth - TIN)], uk, uk + 0.18)))
for i, zsill in enumerate(lv):
    zf = zsill - 0.05
    if i < len(lv) - 1:   # floors 1-2: rear balcony on the party-wall half + side balcony by the corner
        ua, ub = along(uO4, -d4, 0.0, 4.37)
        slab_and_rail(fr4, s4, ua, ub, zf, 0.95, ends=[ua, ub]); balc_zones.append((ua, ub, zsill))
        ua3, ub3 = along(uC3, d3, 0.2, 1.77)
        slab_and_rail(fr3, s3, ua3, ub3, zf, WD, ends=[ua3, ub3], brackets=False)
    else:                 # floor 3: whole rear, past the corner, wrapping 2.98 m down the side
        ua, ub = along(uC4, d4, -WD, abs(uO4 - uC4))
        slab_and_rail(fr4, s4, ua, ub, zf, BD, ends=[ub if d4 > 0 else ua]); balc_zones.append((ua, ub, zsill))
        ua3, ub3 = along(uC3, d3, -BD, 2.98)
        slab_and_rail(fr3, s3, ua3, ub3, zf, WD, ends=[ub3 if d3 > 0 else ua3], brackets=False)
# rust posts down to the ground + white diagonal braces under the side balconies
top_rail = lv[-1] - 0.05 + 1.0
for ue in (along(uC3, d3, -BD, -BD + 0.2), along(uC3, d3, 1.57, 1.77), along(uC3, d3, 2.78, 2.98)):
    adds.append(('brick', fr3.box(ue[0], ue[1], 0.0, top_rail, s3 + WD - 0.2, s3 + WD)))
for zsill in lv:
    zt = zsill - 0.05 - SLAB
    for ue in (along(uC3, d3, 1.57, 1.77), along(uC3, d3, 2.78, 2.98), along(uC3, d3, 0.2, 0.4)):
        adds.append(('white', fr3.side_prism([(s3 - TIN, zt), (s3 + WD, zt), (s3 + WD, zt - 0.15), (s3 - TIN, zt - WD - TIN)], ue[0], ue[1])))
# rear windows: rectangular, Juliet railings where there is no balcony
for o in OPEN['E4']:
    window('E4', o, 'rect', mullion=(o['u1'] - o['u0'] > 1.0), arch_band=False)
    if o.get('garden'):                          # extra mullions on the wide garden door
        for t in (0.25, 0.75):
            uc = o['u0'] + t * (o['u1'] - o['u0'])
            adds.append(('white', fr4.prism(orn.cs_shapely(sbox(uc - FW / 2, o['w0'], uc + FW / 2, o['w1'])), o['s'] - RECESS - TIN, o['s'] - RECESS + 0.1 * MM)))
    if o['w0'] > 3 and not any(a - 0.1 < o['u0'] and o['u1'] < b + 0.1 and abs(z - o['w0']) < 0.5 for a, b, z in balc_zones):
        ua, ub = o['u0'] - 0.1, o['u1'] + 0.1; zf = o['w0']
        adds.append(('white', fr4.box(ua, ub, zf - SLAB, zf, o['s'] - TIN, o['s'] + 0.32)))
        adds.extend(open_rail(fr4, ua, ub, o['s'] + 0.32, zf, h=0.9, side_ends=[ua, ub], s_wall=o['s']))

# ------------------------------------------------------------------ green-grey accents
for fk in ('E1', 'E2'):
    f = FAC[fk]; ops = [o for o in OPEN[fk] if o['w0'] < 14 and o['w0'] > 0.5]
    holes = unary_union([h for k, h in accent_holes if k == fk])
    rows = sorted({round(o['w0'], 1) for o in ops})
    strips = []
    for z in rows:
        same = [o for o in ops if abs(o['w0'] - z) < 0.3]
        strips.append(sbox(f['u0'], z - 0.12 - BAND, f['u1'], z - 0.12))                       # sill course
        sp = max(o['w1'] for o in same) - 0.26
        strips.append(sbox(f['u0'], sp - BAND / 2, f['u1'], sp + BAND / 2))                     # springing course
    if fk == 'E2':
        strips += [sbox(f['u0'], 0.55, f['u0'] + 0.2, 14.3), sbox(f['u1'] - 0.2, 0.55, f['u1'], 14.3)]   # quoins
    accents.append((fk, unary_union(strips).difference(holes), f['s']))
for k, f in FAC.items():
    accents.append((k, sbox(f['u0'] - 0.02, 0.0, f['u1'] + 0.02, 0.55), f['s']))                # plinth
for fk, g, s_w in accents:
    if g.is_empty: continue
    adds.append(('slate', FAC[fk]['fr'].prism(orn.cs_shapely(g), s_w - TIN, s_w + 0.05 * MM)))

# ------------------------------------------------------------------ roof terrace railing on the parapet
line = flat_poly.buffer(-PARA / 2)
if line.geom_type != 'Polygon': line = max(line.geoms, key=lambda p: p.area)
ringp = line.buffer(RT / 2).difference(line.buffer(-RT / 2))
RH = 0.9
adds.append(('dark', orn.to_manifold(trimesh.creation.extrude_polygon(ringp, TOPR)).translate([0, 0, ZFLAT + RH - TOPR])))
Lr = line.exterior.length
for i in range(int(Lr / 0.5)):
    p = line.exterior.interpolate(i * 0.5)
    adds.append(('dark', orn.box(p.x - RT / 2, p.x + RT / 2, p.y - RT / 2, p.y + RT / 2, ZFLAT - 0.05, ZFLAT + RH)))

# ------------------------------------------------------------------ assemble
cut = orn.union(cuts)
body = body - cut; roof = roof - cut
m.add('brick', body); m.add('slate', roof)
for col, s_ in GABLE: m.add(col, s_)
for col, s in adds:
    if col == 'slate': s = s - cut
    m.add(col, s)
parts, summary = orn.finish(m, S, os.path.join(ROOT, 'out', NAME), ring_key='slate')
print(json.dumps({k: summary[k] for k in ['size_mm', 'scale', 'dropped_slivers']}), json.dumps(summary['parts'].get('_exterior'))[:300])
