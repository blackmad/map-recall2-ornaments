"""Da Costakade 13 -> ornament, from the game GLB (models/da-costakade-13.glb)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
import orn, glbsrc, glbbuild
from glbbuild import Frame

NAME, LETTERS, LIT = orn.variant_name('da-costakade-13')
PALETTE = [('brick', '6B4034'), ('white', 'EEEAE0'), ('dark', '27303A'), ('slate', '4E5459')]
ROLES = dict(brick='mass', slate='roof', glass='opening', red='opening', white='trim', stone='trim', concrete='trim',
             dark='drop', bronze='rail', green='drop')
COLOUR = dict(brick='brick', slate='slate', glass='dark', red='dark', white='white', stone='white', concrete='white', dark='dark', bronze='dark', green='dark')

B = glbbuild.Build(NAME, os.path.join(ROOT, 'work/da-costakade-13.raw.glb'), ROLES, PALETTE, COLOUR,
                   float(os.environ.get('ORN_HEIGHT_MM', 60)), lit=LIT, res=0.03)
print(f'scale 1:{1000 / B.S:.0f}, 1 mm = {B.MM:.3f} m')
B.massing()
inl = B.openings(['glass', 'red'])
bal = B.balconies(['bronze'], ['white', 'stone', 'concrete'], 'dark')
zones = [z[1] for z in bal if z[0] == '__balcony_zone__']
def in_zone(c):
    ctr = c.bounds.mean(0)
    for fr, u0, u1, zlo, zhi, s_w in zones:
        P = fr.uws(ctr[None])[0]
        if u0 - 0.1 <= P[0] <= u1 + 0.1 and zlo - 0.1 <= P[1] <= zhi + 0.1 and P[2] > s_w - 0.1: return True
    return False

m = B.model
m.set_priority(['brick', 'slate', 'dark', 'white'])
m.add('brick', B.body)
m.add('slate', B.roof)
for col, s in inl: m.add('white' if col == '__pane__' else col, s)
for col, s in bal:
    if col != '__balcony_zone__': m.add(col, s)
n_trim = n_skip = 0
for k, r in ROLES.items():
    if r != 'trim' or k not in B.g: continue
    for c in glbsrc.components(B.g[k]):
        if in_zone(c): n_skip += 1; continue
        for col, s in B.trim_piece(c, COLOUR[k]): m.add(col, s); n_trim += 1
for k, c in B.mass_details:      # brick/slate pieces off the footprint (corbels): keep unless inside a balcony
    if in_zone(c): n_skip += 1; continue
    for col, s in B.trim_piece(c, COLOUR[k]): m.add(col, s)
print('trim pieces', n_trim, 'skipped in balconies', n_skip, 'balconies', len(zones), 'mass details', len(B.mass_details))
if LIT:
    m.palette = [p for p in PALETTE]
parts, summary = orn.finish(m, B.S, os.path.join(ROOT, 'out', NAME), ring_key='slate')
print(json.dumps({k: summary[k] for k in ['size_mm', 'scale', 'dropped_slivers']}), json.dumps(summary['parts'].get('_exterior')))
