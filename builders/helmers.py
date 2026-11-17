"""Eerste Helmersstraat 148 (pand 0363100012091705, 1899) -> ornament, modelled from scratch.

Evidence: rectified 2021 panorama ref/0363100012091705/front.jpg (50 px/m), thumb.jpg, PDOK ortho,
3DBAG LoD2.2 (roof 50p 15.46 m, max 17.6 m), BAG footprint 4.96 m front x 14.45 m deep.
Ornament edits: depth trimmed to 9 m; bays re-centred on the 4.96 m wall (crop was offset ~0.4 m);
cornice gets a 45-degree cove instead of a flat soffit; balconettes become solid dark balustrades on
chamfered slabs; the diagonal hoisting beam is dropped; rear windows are invented (no photo), plain.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import orn
from glbbuild import Frame

NAME, LETTERS, LIT = orn.variant_name('eerste-helmersstraat-148')
PALETTE = [('brick', '8C4A3A'), ('white', 'F1EEE7'), ('dark', '2B3440'), ('roof', '3C3F44')]
W, D = 4.96, 9.0
TOP = 15.45
S = float(os.environ.get('ORN_HEIGHT_MM', 60)) / TOP
MM = 1 / S
TIN, RECESS, PANE, WALL = 0.8 * MM, 0.4 * MM, 0.45 * MM, 1.6 * MM
FW = max(0.14, 0.5 * MM)        # printable white frame width
DX = (W - 4.13) / 2              # re-centre bays

m = orn.Model(NAME, PALETTE)
m.set_priority(['brick', 'roof', 'dark', 'white'])

# ------------------------------------------------------------------ massing
EAVE = 13.95
body = orn.box(0, W, 0, D, 0, EAVE)
roof_sec = [(0.35, EAVE), (1.40, 15.40), (D, 15.40), (D, EAVE), (0.35, EAVE)]   # front mansard slope, flat behind
roof_sec = [(0.0, EAVE - 0.01), (0.35, EAVE - 0.01), (1.40, 15.40), (D - 0.3, 15.40), (D, EAVE), (D, EAVE - 0.01)]
roofm = orn.yz_prism(roof_sec, 0, W)
# dormer: brick, flush with the facade, low pitched top
DX0, DX1 = 1.17 + DX, 2.95 + DX
dormer = orn.xz_prism([(DX0, 12.9), (DX1, 12.9), (DX1, 15.2), ((DX0 + DX1) / 2, 15.45), (DX0, 15.2)], 0, 1.6)
dormer_roof = orn.yz_prism([(0.0, 15.2), (1.6, 15.2), (1.6, 15.42), (0.0, 15.42)], DX0, DX1) ^ orn.xz_prism([(DX0, 15.0), (DX1, 15.0), (DX1, 15.2), ((DX0 + DX1) / 2, 15.45), (DX0, 15.2)], 0.12, 1.6)
mass = orn.union([body, roofm, dormer])
mass_extra = []
roof_skin = roofm - orn.yz_prism([(0.0, EAVE - 1), (0.35, EAVE - 1), (1.40 - 0.25, 15.40 - 0.9 * MM), (D - 0.3, 15.40 - 0.9 * MM), (D, EAVE - 1)], -1, W + 1)
roof_skin = roof_skin - orn.box(DX0, DX1, -1, 1.6, 0, 20)
cuts, inl = [], []
front = Frame([0, -1, 0]); back = Frame([0, 1, 0])

def window(fr, s_w, x0, x1, z0, z1, arch=False, transom=True, mullion=False, uflip=False):
    """Window opening on a facade frame. x is building x; for the back facade u = -x."""
    u0, u1 = (-x1, -x0) if uflip else (x0, x1)
    poly = orn.arch_poly(u0, u1, z0, z1 - (u1 - u0) / 2) if arch else [(u0, z0), (u1, z0), (u1, z1), (u0, z1)]
    if LIT:
        cuts.append(fr.prism(poly, s_w - 3, s_w + 3))
        inl.append(('white', fr.prism(poly, s_w - RECESS - PANE, s_w - RECESS)))
    else:
        cuts.append(fr.prism(poly, s_w - RECESS, s_w + 3))
        inl.append(('dark', fr.prism(poly, s_w - RECESS - TIN, s_w - RECESS)))
    # white frame ring, flush with the wall face, keyed back into the glass plane
    outer = orn.cs_of(poly)
    ring = outer - outer.offset(-FW, join_type=orn.JoinType.Miter) if hasattr(orn, 'JoinType') else outer
    inl.append(('white', fr.prism(ring, s_w - RECESS - TIN, s_w - 0.05 * MM)))
    bars = []
    if transom and not arch and z1 - z0 > 1.4:
        zt = z1 - 0.45
        bars.append([(u0, zt - FW / 2), (u1, zt - FW / 2), (u1, zt + FW / 2), (u0, zt + FW / 2)])
    if mullion:
        uc = (u0 + u1) / 2
        bars.append([(uc - FW / 2, z0), (uc + FW / 2, z0), (uc + FW / 2, z1), (uc - FW / 2, z1)])
    for b in bars: inl.append(('white', fr.prism(b, s_w - RECESS - TIN, s_w - RECESS + 0.1 * MM)))
    # sill
    inl.append(('white', fr.prism([(u0 - 0.06, z0 - 0.12), (u1 + 0.06, z0 - 0.12), (u1 + 0.06, z0), (u0 - 0.06, z0)], s_w - TIN, s_w + 0.25 * MM)))

def keystone(xc, z):
    k = max(0.22, 0.6 * MM)
    inl.append(('white', front.prism([(xc - k / 2, z), (xc + k / 2, z), (xc + k / 2, z + k), (xc - k / 2, z + k)], -TIN, 0.3 * MM)))

L = (0.0 + DX, 1.02 + DX); M = (1.50 + DX, 2.57 + DX); R = (3.05 + DX, 4.13 + DX)
# ground floor
window(front, 0, *L, 1.0, 3.25)
window(front, 0, *M, 0.95, 3.25)
DOOR_D = 0.25 if LIT else 0.6                     # portal depth (user 2026-11-17: door more inset)
DX0d, DX1d = R[0], R[1] - 0.05
cuts.append(orn.box(DX0d, DX1d, -1, DOOR_D, 0.0, 3.30))
door_back = WALL if LIT else DOOR_D + TIN
inl.append(('dark', orn.box(DX0d, DX1d, DOOR_D, door_back, 0.0, 2.55)))           # door leaf
inl.append(('white', orn.box(DX0d, DX1d, DOOR_D, door_back, 2.55, 2.55 + FW)))    # transom bar
inl.append(('white' if LIT else 'dark', orn.box(DX0d, DX1d, DOOR_D + (0 if LIT else 0.05), door_back, 2.55 + FW, 3.30)))  # transom light
door_ring = orn.cs_of([(DX0d - FW, 0.0), (DX1d + FW, 0.0), (DX1d + FW, 3.30 + FW), (DX0d - FW, 3.30 + FW)]) - orn.cs_of([(DX0d, -1), (DX1d, -1), (DX1d, 3.30), (DX0d, 3.30)])
inl.append(('white', front.prism(door_ring, -TIN, 0.25 * MM)))
# floor 1
window(front, 0, *L, 4.10, 6.05); window(front, 0, *M, 3.68, 6.05, mullion=True); window(front, 0, *R, 4.10, 6.05)
# floor 2
window(front, 0, *L, 7.00, 8.90); window(front, 0, *M, 6.55, 8.90, mullion=True); window(front, 0, *R, 7.00, 8.90)
# floor 3: arched centre window with a carved tympanum
window(front, 0, *L, 9.80, 11.70); window(front, 0, *R, 9.80, 11.75)
window(front, 0, *M, 9.80, 12.30, arch=True)
r = (M[1] - M[0]) / 2
tymp = orn.arch_poly(M[0] + FW, M[1] - FW, 11.70, 11.75, 20)
inl.append(('white', front.prism(tymp, -RECESS - TIN, 0.15 * MM)))
# dormer window
window(front, 0, 1.50 + DX, 2.57 + DX, 13.95, 14.65, transom=False)
# keystones above the side windows (and above ground-floor windows)
for z, xs in [(3.45, [L, M]), (6.25, [L, M, R]), (9.10, [L, R]), (11.95, [L, R])]:
    for x in xs: keystone((x[0] + x[1]) / 2, z)

# balconettes (user 2026-11-17: open, so they read as balconies): a white slab with a 45-degree
# underside carries three thin dark railing walls; the space inside shows the French door behind.
P = 1.8 * MM
RT = 0.6 * MM
for zf, zt in [(3.68, 4.55), (6.55, 7.45)]:
    x0, x1 = M[0] - 0.12, M[1] + 0.12
    slab_t = max(0.12, 0.4 * MM)
    inl.append(('white', orn.box(x0, x1, -P, TIN, zf - slab_t, zf)))
    inl.append(('white', orn.yz_prism([(-P, zf - slab_t + 0.01), (-P, zf - slab_t), (TIN, zf - slab_t - (P + TIN)), (TIN, zf - slab_t + 0.01)], x0, x1)))
    KICK, TOPR = max(0.1, 0.4 * MM), max(0.09, 0.4 * MM)
    rails = []
    for (a0, a1, b0, b1) in [(x0, x1, -P, -P + RT), (x0, x0 + RT, -P, 0.02), (x1 - RT, x1, -P, 0.02)]:
        rails.append(orn.box(a0, a1, b0, b1, zf, zf + KICK))            # kick plate
        rails.append(orn.box(a0, a1, b0, b1, zt - TOPR, zt))            # hand rail
    n_post = 5
    for i in range(n_post):                                             # front balusters
        xp = x0 + (x1 - x0 - RT) * i / (n_post - 1)
        rails.append(orn.box(xp, xp + RT, -P, -P + RT, zf, zt))
    for xp in (x0, x1 - RT):                                            # back posts against the wall
        rails.append(orn.box(xp, xp + RT, -RT, 0.02, zf, zt))
    for r_ in rails: inl.append(('dark', r_))

# cornice: white, 0.55 m proud with a 45-degree cove, split by the dormer window
CP, CH = 0.28, 0.32                 # slimmer eaves board (user 2026-11-17: white blocks too much)
cor = [(0.02, EAVE), (-CP, EAVE), (-CP, EAVE - CH), (0.02, EAVE - CH - CP - 0.02)]
for a, b in [(0, 1.50 + DX), (2.57 + DX, W)]:
    inl.append(('white', orn.yz_prism(cor, a, b)))
# dormer lintel
inl.append(('white', front.prism([(1.42 + DX, 14.65), (2.65 + DX, 14.65), (2.65 + DX, 14.85), (1.42 + DX, 14.85)], -TIN, 0.3 * MM)))
# brick dentil band suggestion: a slight brick string course under the cornice is hidden by the cove; skip

# rear facade (no photo: plain 3-bay sash windows, invented)
for z0, z1 in [(4.1, 6.05), (7.0, 8.9), (9.8, 11.7)]:
    for x in (L, M, R): window(back, D, *x, z0, z1, transom=False, uflip=True)
window(back, D, *R, 1.0, 3.0, transom=False, uflip=True)      # beside the extension

# kitchen extension (BAG footprint's rear annex; user 2026-11-17 aerial): one storey, flat bitumen roof
# with a ridge skylight (rectangular footprint, triangular section), wide glazed door at the back
AX0, AX1, AD, AH = 0.0, W - 1.25, 4.0, 3.3
annex = orn.box(AX0, AX1, D - 0.05, D + AD, 0, AH)
mass_extra = [annex]
inl.append(('roof', orn.box(AX0, AX1, D - 0.05, D + AD, AH - 0.8 * MM, AH)))
sk_c, sk_w = (AX0 + AX1) / 2, 1.3
skylight = orn.yz_prism([(D + 1.0, AH - 0.05), (D + 3.1, AH - 0.05), (D + 3.1, AH), (D + 1.0, AH)], sk_c - sk_w / 2, sk_c + sk_w / 2)
skylight = orn.xz_prism([(sk_c - sk_w / 2, AH - 0.05), (sk_c + sk_w / 2, AH - 0.05), (sk_c, AH + sk_w / 2)], D + 1.0, D + 3.1)
inl.append(('white' if LIT else 'dark', skylight))
window(back, D + AD, AX0 + 0.45, AX1 - 0.45, 0.15, 2.6, transom=False, mullion=True, uflip=True)

mass = orn.union([mass] + mass_extra) if mass_extra else mass
cut = orn.union(cuts)
mass = mass - cut
roof_skin = roof_skin - cut
if LIT:
    # cavity: box with a 45-degree gabled top (along y) so the ceiling never bridges
    hw = W / 2 - WALL
    zc = min(15.40 - WALL - hw, EAVE)
    cav = orn.yz_prism([(WALL, -1), (D - WALL, -1), (D - WALL, 1)], 0, 1)   # placeholder
    sec = [(WALL, -WALL), (W - WALL, -WALL), (W - WALL, zc), (W / 2, zc + hw), (WALL, zc)]
    cav = orn.xz_prism(sec, WALL, D - WALL)
    mass = mass - cav; roof_skin = roof_skin - cav
m.add('brick', mass)
m.add('roof', roof_skin)
m.add('roof', dormer_roof)
for col, s in inl: m.add(col, s)

parts, summary = orn.finish(m, S, os.path.join(ROOT, 'out', NAME), ring_key='roof')
print(json.dumps({k: summary[k] for k in ['size_mm', 'scale', 'dropped_slivers']}), json.dumps(summary['parts'].get('_exterior')))
