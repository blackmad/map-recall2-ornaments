"""Export out/*/ STLs to demo/models/<name>.json (base64 float32 verts + uint32 faces per colour) and meta.json."""
import trimesh, glob, os, json, base64, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'demo'); os.makedirs(f'{D}/models', exist_ok=True)
meta = {}
for d in sorted(glob.glob(f'{ROOT}/out/*/')):
    n = os.path.basename(d.rstrip('/')); parts = []
    BOAT_Z = 2.0 if n.endswith('-model') else 1.2
    if n.endswith('-boat'): continue                   # shown moored inside its houseboat's model
    for f in sorted(glob.glob(f'{d}/*.stl')) + sorted(glob.glob(f'{d}/base/*.stl')) + sorted(glob.glob(f'{d.rstrip("/")}-boat/*.stl')):
        k, h = os.path.basename(f)[:-4].split('__')[1:]
        m = trimesh.load(f)
        if '/base/' in f: k = 'base'; m.apply_translation([0, 0, -12])     # exploded view: the snap bottom below
        if '-boat/' in f: k = 'boat-' + k; m.apply_translation([0, 0, BOAT_Z])  # moored on its pins (water plate thickness)
        parts.append(dict(k=k, hex=h, p=base64.b64encode(m.vertices.astype('<f4').tobytes()).decode(),
                          i=base64.b64encode(m.faces.astype('<u4').tobytes()).decode()))
    json.dump(dict(parts=parts), open(f'{D}/models/{n}.json', 'w'))
    rep = json.load(open(f'{d}/{n}-report.json'))
    meta[n] = dict(size=rep['size_mm'], scale=rep['scale'], colours=rep['extruders'], overhang=round(rep['parts']['_exterior']['overhang_mm2']))
json.dump(meta, open(f'{D}/meta.json', 'w'), indent=1)
print('exported', len(meta))
