"""Export out/*/ STLs to demo/models/<name>.json (base64 float32 verts + uint32 faces per colour) and meta.json."""
import trimesh, glob, os, json, base64, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'demo'); os.makedirs(f'{D}/models', exist_ok=True)
meta = {}
for d in sorted(glob.glob(f'{ROOT}/out/*/')):
    n = os.path.basename(d.rstrip('/')); parts = []
    BOAT_Z = 2.0 if '-model' in n else 1.2
    if n.endswith('-boat') or n.endswith('-wheel'): continue   # shown moored inside its houseboat's model
    for f in sorted(glob.glob(f'{d}/*.stl')) + sorted(glob.glob(f'{d}/base/*.stl')) + sorted(glob.glob(f'{d.rstrip("/")}-boat/*.stl')) + sorted(glob.glob(f'{d.rstrip("/")}-wheel/*.stl')):
        k, h = os.path.basename(f)[:-4].split('__')[1:]
        m = trimesh.load(f)
        if '/base/' in f: k = 'base'; m.apply_translation([0, 0, -12])     # exploded view: the snap bottom below
        if '-boat/' in f: k = 'boat-' + k; m.apply_translation([0, 0, BOAT_Z])  # moored on its pins (water plate thickness)
        if '-wheel/' in f:                                                   # steering wheel on its console axle
            import numpy as np
            k = 'wheel-' + k; m.apply_transform(np.array(json.load(open(os.path.join(os.path.dirname(f), 'pose.json')))['matrix']))
            m.apply_translation([0, 0, BOAT_Z])
        sw = os.path.join(os.path.dirname(f), 'swap.json')
        if k == 'bluegrey' and os.path.exists(sw):   # one slot, spool swapped mid-print: show blue below, grey above
            sj = json.load(open(sw)); z = sj['swap_z_mm']
            pieces = [('blue', sj['below_hex'], m.slice_plane([0, 0, z], [0, 0, -1], cap=True)),
                      ('grey', sj['above_hex'], m.slice_plane([0, 0, z], [0, 0, 1], cap=True))]
        else:
            pieces = [(k, h, m)]
        for k, h, m in pieces:
            parts.append(dict(k=k, hex=h, p=base64.b64encode(m.vertices.astype('<f4').tobytes()).decode(),
                              i=base64.b64encode(m.faces.astype('<u4').tobytes()).decode()))
    json.dump(dict(parts=parts), open(f'{D}/models/{n}.json', 'w'))
    rep = json.load(open(f'{d}/{n}-report.json'))
    meta[n] = dict(size=rep['size_mm'], scale=rep['scale'], colours=rep['extruders'], overhang=round(rep['parts']['_exterior']['overhang_mm2']))
json.dump(meta, open(f'{D}/meta.json', 'w'), indent=1)
print('exported', len(meta))
