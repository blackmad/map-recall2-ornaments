"""Woonark Jacob van Lennepkade 520 -> ornament (~10 cm, hanging) or model (~23 cm, no hanger), plus a detachable sloop.

Sources: BAG ligplaats 0363020001035889 (20.4 x 5.5 m mooring), PDOK 8 cm orthophoto rectified along the boat axis
(ref/ark520/rect.png: roof 19.4 x 5.0 m, solar banks at the SW end, dome skylight NE, sunken terrace mid canal side,
deck + pontoon at the NE end, sloop moored across the NE end), and the user's street/canal photos (2026-11-17):
white vertical-plank walls on a grey concrete hull, dark roof fascia, brown window frames, a glazed NE corner
(sliding doors) with a white lounge chair visible inside, plant pots and vents on the roof.

Local frame: x along the boat from the SW end, y across from the canal side (y=0) to the quay side, z up from the
waterline. ORN_SIZE=ornament|model. Outputs out/ark-520[-model]/ and out/ark-520[-model]-boat/.
"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np, trimesh
from shapely.geometry import Polygon as SP, box as sbox, Point
import orn
from orn import box

SIZE = os.environ.get('ORN_SIZE', 'ornament')
MIX = os.environ.get('ORN_MIX', '0') == '1'        # 5 slots: blue water AND brown frames; grey is mixed from white+black by the user
BLUE = MIX or os.environ.get('ORN_BLUE', '0') == '1'  # blue water (and solar panels); without MIX frames/pots/chairs go charcoal
BOAT = os.environ.get('ORN_NOBOAT', '0') != '1'    # house only: no sloop, no pins, water trimmed to the ark
NAME = 'ark-520' + ('' if SIZE == 'ornament' else '-model') + ('-mix' if MIX else '-blue' if BLUE else '') + ('' if BOAT else '-noboat')
PALETTE = ([('white', 'EEEBE3'), ('grey', '8E8E88'), ('charcoal', '2A2C30'), ('brown', '4B2E24'), ('blue', '3D6F9C')] if MIX else
           [('white', 'EEEBE3'), ('grey', '8E8E88'), ('charcoal', '2A2C30'), ('blue', '3D6F9C')] if BLUE else
           [('white', 'EEEBE3'), ('grey', '8E8E88'), ('charcoal', '2A2C30'), ('brown', '4B2E24')])
def C(c):
    """Logical colour -> palette key for this variant."""
    if c == 'brown' and BLUE and not MIX: return 'charcoal'
    if c in ('water', 'solar'): return 'blue' if BLUE else 'charcoal'
    return c
PX0, PX1, PY0, PY1 = -0.6, 23.4, -1.9, 5.6          # water plate (m)
LEN_MM = 100.0 if SIZE == 'ornament' else 230.0
S = LEN_MM / (PX1 - PX0); MM = 1.0 / S
print(f'{NAME}: scale 1:{1000 / S:.0f}, 1 mm = {MM:.3f} m')
FW = max(0.09, 0.5 * MM)          # frame / mullion bar
RECESS, TIN = 0.35 * MM, 0.8 * MM
PLATE = (1.2 if SIZE == 'ornament' else 2.0) * MM

# ---------------------------------------------------------------- dimensions (m)
WX0, WX1, WY0, WY1 = 0.1, 19.3, 0.1, 4.9           # wall faces
ZH, ZW, ZR = 0.65, 2.95, 3.3                        # hull top, wall top / fascia bottom, roof top
OVH = 0.15                                          # roof overhang
TX0, TX1, TY1, ZT = 8.9, 12.4, 3.4, 2.0             # sunken terrace (canal side) and its floor
m = orn.Model(NAME, PALETTE)
m.set_priority([k for k, _ in PALETTE])   # later wins (white < grey < charcoal < brown/blue)
P = []                                              # (colour, solid)

def fbox(face, u0, u1, z0, z1, d0, d1):
    """Box on a wall face: u along the face, d inward depth from the face (negative = proud)."""
    if face == 'S': return box(u0, u1, WY0 + d0, WY0 + d1, z0, z1)
    if face == 'N': return box(u0, u1, WY1 - d1, WY1 - d0, z0, z1)
    if face == 'W': return box(WX0 + d0, WX0 + d1, u0, u1, z0, z1)
    if face == 'E': return box(WX1 - d1, WX1 - d0, u0, u1, z0, z1)

# hull, walls, roof
hull = box(WX0 - 0.06, WX1 + 0.06, WY0 - 0.06, WY1 + 0.06, 0.0, ZH)
walls = box(WX0, WX1, WY0, WY1, ZH - 0.05, ZW + 0.05)
roof = box(WX0 - OVH, WX1 + OVH, WY0 - OVH, WY1 + OVH, ZW, ZR)
# sunken terrace: everything above ZT goes, lowered roof skin + fascia on the canal side
cut_t = box(TX0, TX1, -5, TY1, ZT, 10)
walls = walls - cut_t; roof = roof - cut_t
P.append(('charcoal', box(TX0, TX1, WY0 - OVH, TY1, ZT - 0.9 * MM, ZT)))
P.append(('charcoal', box(TX0, TX1, WY0 - 0.12, WY0 + 0.25, ZT - 0.25, ZT + 0.05)))
# plank grooves in the white walls (vertical boards)
GS = max(0.16, 1.2 * MM); GW = max(0.03, 0.3 * MM); GD = max(0.02, 0.2 * MM)
grooves = []
for face, a, b in (('S', WX0, WX1), ('N', WX0, WX1), ('W', WY0, WY1), ('E', WY0, WY1)):
    u = a + GS
    while u < b - 0.05:
        grooves.append(fbox(face, u, u + GW, ZH + 0.05, ZW - 0.02, -0.01, GD)); u += GS
walls = walls - orn.union(grooves)

# ---------------------------------------------------------------- windows
cuts = []
def window(face, u0, u1, z0, z1, panes=1, open_=False, door=False):
    cuts.append(fbox(face, u0, u1, z0, z1, -0.05, RECESS if not open_ else 3.0))
    if door:
        P.append(('brown', fbox(face, u0, u1, z0, z1, RECESS, RECESS + TIN))); return
    if not open_:
        P.append(('grey', fbox(face, u0, u1, z0, z1, RECESS, RECESS + TIN)))     # glass with drawn curtains
    ring = [fbox(face, u0, u0 + FW, z0, z1, 0.0, RECESS + TIN), fbox(face, u1 - FW, u1, z0, z1, 0.0, RECESS + TIN),
            fbox(face, u0, u1, z0, z0 + FW, 0.0, RECESS + TIN), fbox(face, u0, u1, z1 - FW, z1, 0.0, RECESS + TIN)]
    for i in range(1, panes):
        uc = u0 + (u1 - u0) * i / panes
        ring.append(fbox(face, uc - FW / 2, uc + FW / 2, z0, z1, 0.0, RECESS + TIN))
    for r_ in ring: P.append(('brown', r_))

# canal side (S), SW -> NE
window('S', 0.3, 2.6, 1.0, 2.7, panes=3); window('W', 0.3, 1.3, 1.0, 2.7)          # SW corner window
window('S', 4.2, 5.0, 1.25, 2.6); window('S', 5.9, 7.6, 1.25, 2.6, panes=2)
window('S', 9.4, 11.2, 0.95, 1.65, panes=2)                                         # under the sunken terrace
for a, b in ((12.7, 14.3), (14.3, 15.9), (15.9, 17.4)):
    window('S', a, b, 0.85, 2.75, panes=1)
window('S', 17.4, 19.15, 0.85, 2.75, open_=True)                                    # open NE corner bay
window('E', 0.25, 1.4, 0.85, 2.75, open_=True)
# quay side (N)
window('N', 2.0, 3.0, 1.3, 2.5); window('N', 4.4, 5.5, 1.3, 2.5)
window('N', 7.0, 8.2, 1.85, 2.45, panes=2); window('N', 8.6, 9.5, 0.7, 2.65, door=True)
window('N', 10.3, 11.3, 1.3, 2.5); window('N', 15.5, 16.3, 1.4, 2.4)
window('E', 2.2, 3.1, 0.7, 2.65, door=True)                                         # door to the NE pontoon
# the NE corner room behind the open glazing: dark-lined, white lounge chair just inside the corner
RX0, RY1, RZ0, RZ1 = 16.9, 3.0, ZH + 0.12, ZW - 0.12
RIN = 0.32                        # wall thickness around the room
room = box(RX0, WX1 - RIN, WY0 + RIN, RY1, RZ0, RZ1)
cuts.append(room)
LIN = max(0.1, 0.6 * MM)
lining = box(RX0 - LIN, WX1 - RIN + LIN, WY0 + RIN - LIN, RY1 + LIN, RZ0 - LIN, RZ1 + LIN) - room
cx0, cy0 = 18.2, WY0 + RIN + 0.08  # chair footprint corner (x, y); its back is to the canal-side glass (user)
chair = [box(cx0, cx0 + 0.75, cy0, cy0 + 0.75, RZ0 - 0.01, RZ0 + 0.42),                   # seat block
         box(cx0, cx0 + 0.75, cy0, cy0 + 0.2, RZ0 - 0.01, RZ0 + 0.95),                     # back, window side
         box(cx0, cx0 + 0.14, cy0, cy0 + 0.75, RZ0 - 0.01, RZ0 + 0.62),                    # arms
         box(cx0 + 0.61, cx0 + 0.75, cy0, cy0 + 0.75, RZ0 - 0.01, RZ0 + 0.62)]
cut = orn.union(cuts)
walls = walls - cut; hull = hull - cut
P.append(('charcoal', lining - cut))
P.append(('white', orn.union(chair)))

# ---------------------------------------------------------------- roof furniture
def tilted_bank(x0, x1, y0, y1, nx, ny, rise=0.3):
    poly = [(y0, ZR - 0.02), (y1, ZR - 0.02), (y1, ZR + 0.08 + rise), (y0, ZR + 0.08)]
    panel = orn.yz_prism(poly, x0, x1); bars = []
    bar_t = max(0.06, 0.3 * MM)
    def zt(y): return ZR + 0.08 + rise * (y - y0) / (y1 - y0)
    for i in range(nx + 1):          # cross bars along y at panel joints
        xb = x0 + (x1 - x0) * i / nx; xa = min(max(xb - FW / 2, x0), x1 - FW)
        bars.append(orn.yz_prism([(y0, zt(y0) - 0.06), (y1, zt(y1) - 0.06), (y1, zt(y1) + bar_t), (y0, zt(y0) + bar_t)], xa, xa + FW))
    for j in range(ny + 1):
        yb = y0 + (y1 - y0) * j / ny; ya = min(max(yb - FW / 2, y0), y1 - FW)
        bars.append(orn.yz_prism([(ya, zt(ya) - 0.06), (ya + FW, zt(ya + FW) - 0.06), (ya + FW, zt(ya + FW) + bar_t), (ya, zt(ya) + bar_t)], x0, x1))
    bars = orn.union(bars)
    P.append(('solar', panel - bars)); P.append(('white', bars))   # grid sunk into the panel so it cannot float
tilted_bank(0.75, 5.9, 0.25, 2.25, 3, 2)
tilted_bank(0.75, 5.33, 3.4, 4.4, 3, 1, rise=0.18)
# dome skylight: white upstand + dome
P.append(('white', box(15.5, 16.9, 2.7, 4.1, ZR - 0.02, ZR + 0.15)))
dome = trimesh.creation.icosphere(subdivisions=3, radius=1.0); dome.apply_scale([0.62, 0.62, 0.32]); dome.apply_translation([16.2, 3.4, ZR + 0.15])
P.append(('white', orn.to_manifold(dome) ^ box(15, 18, 2, 5, ZR + 0.14, 10)))
# planters along the quay edge, vents
def pot(x, y, d=0.55, h=0.32):
    c = trimesh.creation.cylinder(radius=d / 2, height=h, sections=24); c.apply_translation([x, y, ZR + h / 2 - 0.02])
    return orn.to_manifold(c)
for x in (6.6, 7.6, 9.9, 12.9, 13.7, 14.5):
    P.append(('brown', pot(x, 4.45)))
for x, y in ((7.0, 3.5), (13.2, 0.9), (15.0, 4.3)):
    P.append(('grey', pot(x, y, d=max(0.28, 1.3 * MM), h=0.3)))
# ladder from the terrace up to the roof, against the quay-side wall of the terrace
LR = max(0.07, 0.45 * MM)
for xl in (11.4, 11.85 - LR):
    P.append(('grey', box(xl, xl + LR, TY1 - 0.22, TY1 - 0.22 + LR, ZT - 0.01, ZR + 0.5)))
for zr in np.arange(ZT + 0.3, ZR + 0.4, 0.3):
    P.append(('grey', box(11.4, 11.85, TY1 - 0.22, TY1 - 0.22 + LR, zr, zr + LR)))

# ---------------------------------------------------------------- decks, chair, pontoon, water
DZ = 0.45
deck = box(13.3, 18.6, -1.45, WY0 - 0.06, 0.0, DZ)
plank = max(0.2, 1.2 * MM)
for yy in np.arange(-1.45 + plank, WY0 - 0.1, plank):
    deck = deck - box(13.2, 18.7, yy, yy + max(0.03, 0.3 * MM), DZ - max(0.02, 0.2 * MM), DZ + 1)
P.append(('grey', deck))
# Adirondack chair on the deck (brown): slanted seat + back, on a solid base so it prints
ax, ay = 14.4, -1.15
P.append(('brown', orn.yz_prism([(ay, DZ - 0.01), (ay + 0.75, DZ - 0.01), (ay + 0.75, DZ + 0.95), (ay + 0.6, DZ + 0.95), (ay + 0.5, DZ + 0.38), (ay, DZ + 0.42)], ax, ax + 0.7)))
P.append(('grey', orn.to_manifold(trimesh.creation.cylinder(radius=0.32, height=0.06, sections=24).apply_translation([16.0, -0.7, DZ + 0.72])) +
          box(15.97, 16.03, -0.73, -0.67, DZ - 0.01, DZ + 0.72) + box(15.75, 16.25, -0.95, -0.45, DZ - 0.01, DZ + 0.04)))
pont = box(WX1 + 0.12, 20.6, -0.95, 2.6, 0.0, 0.4)
P.append(('grey', pont))
for x, y in ((20.1, -0.45), (20.15, 0.35)):
    c = trimesh.creation.cylinder(radius=0.24, height=0.42, sections=20); c.apply_translation([x, y, 0.4 + 0.2])
    P.append(('brown', orn.to_manifold(c)))
water_poly = sbox(PX0, PY0, PX1, PY1).buffer(-0.8).buffer(0.8, quad_segs=8)
if not BOAT:   # just the ark: water hugs the hull, deck and pontoon
    water_poly = sbox(WX0 - 0.06, -1.45, 20.6, WY1 + 0.06).buffer(0.55, join_style=2).buffer(-0.4).buffer(0.4, quad_segs=8)
water = orn.to_manifold(trimesh.creation.extrude_polygon(water_poly, PLATE)).translate([0, 0, -PLATE])
# sloop pins (the boat plugs onto them); mm dimensions are absolute
PIN_D, PIN_H, HOLE_D, HOLE_H = 1.8, 1.3, 2.0, 1.5
BX, BY0, BY1, BEAM = 21.85, -0.3, 4.7, 1.9
PINS = [(BX, 0.7), (BX, 2.6)] if BOAT else []
pins = []
for x, y in PINS:
    c = trimesh.creation.cylinder(radius=PIN_D / 2 * MM, height=PIN_H * MM + 0.02, sections=24)
    c.apply_translation([x, y, (PIN_H * MM) / 2 - 0.01]); pins.append(orn.to_manifold(c))
P.append(('water', water + orn.union(pins) if pins else water))
P.append(('grey', hull)); P.append(('white', walls)); P.append(('charcoal', roof))
for col, s_ in P: m.add(C(col), s_)

if BOAT:
    # ---------------------------------------------------------------- the sloop (separate print)
    b = orn.Model(NAME + '-boat', PALETTE)
    b.set_priority([k for k, _ in PALETTE])
    plan = SP([(BX - BEAM / 2, BY0), (BX + BEAM / 2, BY0), (BX + BEAM / 2, 3.1), (BX + 0.5, 4.25), (BX, BY1), (BX - 0.5, 4.25), (BX - BEAM / 2, 3.1)])
    plan = plan.buffer(-0.12).buffer(0.12)
    FB = 0.8                                           # freeboard
    hull_b = []
    NF = 20
    for i in range(NF):                                # flared hull: bottom inset 0.25 m, rounded bilge
        z0, z1 = FB * 0.4 * i / NF, FB * 0.4 * (i + 1) / NF
        t = (i + 0.5) / NF
        hull_b.append(orn.to_manifold(trimesh.creation.extrude_polygon(plan.buffer(-0.25 * (1 - math.sin(t * math.pi / 2))), z1 - z0 + 0.002)).translate([0, 0, z0]))
    hull_b.append(orn.to_manifold(trimesh.creation.extrude_polygon(plan, FB * 0.6)).translate([0, 0, FB * 0.4]))
    hull_b = orn.union(hull_b)
    # open boat (user: uncovered): hollow hull with a floor, stern bench, mid thwart and a small console
    WB = max(0.1, 0.9 * MM)                             # hull wall
    FLOOR = 0.42                                        # floor top (solid below, so the pin holes stay hidden)
    well = plan.buffer(-WB) & sbox(BX - 2, BY0 + WB, BX + 2, 3.3)
    hull_b = hull_b - orn.to_manifold(trimesh.creation.extrude_polygon(well, 2.0)).translate([0, 0, FLOOR])
    seats = orn.union([orn.to_manifold(trimesh.creation.extrude_polygon(well & sbox(BX - 2, y0, BX + 2, y1), 0.22)).translate([0, 0, FLOOR - 0.01])
                       for y0, y1 in ((BY0, BY0 + 0.55), (1.45, 1.85))])
    floor_ = orn.to_manifold(trimesh.creation.extrude_polygon(well, max(0.03, 0.25 * MM))).translate([0, 0, FLOOR - 0.01])
    console = box(BX + 0.12, BX + 0.62, 2.3, 2.75, FLOOR - 0.01, FB + 0.25) - box(BX + 0.1, BX + 0.64, 2.3, 2.5, FB + 0.05, FB + 0.3)
    rub = orn.to_manifold(trimesh.creation.extrude_polygon(plan.buffer(0.05).difference(plan.buffer(-0.08)), 0.1)).translate([0, 0, FB - 0.18])
    motor = box(BX - 0.18, BX + 0.18, BY0 - 0.45, BY0 + 0.02, 0.35, FB + 0.45) + box(BX - 0.07, BX + 0.07, BY0 - 0.3, BY0 - 0.12, 0.0, 0.4)
    holes = []
    for x, y in PINS:
        c = trimesh.creation.cylinder(radius=HOLE_D / 2 * MM, height=HOLE_H * MM * 2, sections=24); c.apply_translation([x, y, 0]); holes.append(orn.to_manifold(c))
    holes = orn.union(holes)
    b.add('white', hull_b - holes); b.add('white', rub); b.add('grey', floor_ + seats); b.add('white', console); b.add('grey', motor - holes)

out = os.path.join(ROOT, 'out', NAME)
# centre of mass, measured on the finished print parts (the pre-resolve estimate was off by ~0.07 m = 4 degrees)
if BOAT:
    bparts, bsum = orn.finish(b, S, os.path.join(ROOT, 'out', NAME + '-boat'), ring=False)
    print('boat', json.dumps({k: bsum[k] for k in ['size_mm', 'scale']}))
def print_com(model_parts, extra=()):
    ms = list(model_parts.values()) + list(extra)
    return sum(t.center_mass * t.volume for t in ms) / sum(t.volume for t in ms) / S   # model metres (finish only shifts z)
if SIZE == 'ornament':
    import copy
    m0 = copy.copy(m); m0.parts = {k: list(v) for k, v in m.parts.items()}
    parts0, _ = orn.finish(m0, S, out, ring=False)
    com = print_com(parts0, bparts.values() if BOAT else ())   # balanced with the sloop moored; -noboat balances without
    print('com', np.round(com, 3))
    # hanger: two 3/4 eyelets sunk into the main roof, on the centre-of-mass line across the boat and symmetric about
    # it along the boat, either side of the sunken terrace. A loop of thread through both hangs the ark level in
    # both axes; a single ring could not stand over the centre of mass without sitting in the terrace pit.
    EOD, ET = 5.5, 1.7                               # eyelet outer diameter / tube (mm): ~2.1 mm hole for thread
    Re, re_ = (EOD - ET) / 2 * MM, ET / 2 * MM
    a = max(com[0] - (TX0 - Re - re_ - 0.25), (TX1 + Re + re_ + 0.25) - com[0])
    for ex in (com[0] - a, com[0] + a):
        t = orn.torus(Re, re_).rotate([90, 0, 90]).translate([ex, com[1], ZR + Re * math.cos(math.radians(45))])
        m.add('charcoal', t)
    print('eyelets at x', round(com[0] - a, 2), round(com[0] + a, 2), 'y', round(com[1], 2))
parts, summary = orn.finish(m, S, out, ring=False)
print(json.dumps({k: summary[k] for k in ['size_mm', 'scale', 'dropped_slivers']}), json.dumps(summary['parts'].get('_exterior'))[:200])

