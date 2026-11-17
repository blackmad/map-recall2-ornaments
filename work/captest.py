import sys, glob, time; sys.path.insert(0,'tools'); import orn, basecap, trimesh, json
for n in sys.argv[1:]:
    t=time.time()
    parts={f.split('__')[1]:trimesh.load(f) for f in glob.glob(f'out/{n}/*.stl')}
    new, cap, info = basecap.apply(parts, orn.union, orn.to_manifold, orn.to_trimesh)
    print(n, round(time.time()-t,1),'s', json.dumps(info))
    if cap is not None:
        print('  cap bodies', len(orn.to_manifold(cap).decompose()), 'bounds', cap.bounds.round(1).tolist(), 'vol', round(cap.volume,1))
        cap.export(f'work/cap_{n}.stl')
        sh = orn.union([orn.to_manifold(p) for p in new.values()]); cm = orn.to_manifold(cap)
        print('  overlap cap/shell mm3', round((sh ^ cm).volume(),3))
        cmA = cm.translate([0, 0, -basecap.PLATE])
        print('  assembled overlap (bead interference, expect ~0)', round((sh ^ cmA).volume(), 3), 'mm3; vs ungrooved shell', round((orn.union([orn.to_manifold(p) for p in parts.values()]) ^ cmA).volume(), 3))
