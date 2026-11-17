# blender -b -P render.py -- input.(glb|3mf-parts dir|stl...) outprefix [up=Y|Z]
import bpy, sys, math, mathutils, os
argv = sys.argv[sys.argv.index('--')+1:]
inp, outp = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
paths = [inp] if not os.path.isdir(inp) else sorted(os.path.join(inp,f) for f in os.listdir(inp) if f.endswith(('.stl','.glb','.ply','.obj')))
for p in paths:
    if p.endswith('.glb'): bpy.ops.import_scene.gltf(filepath=p)
    elif p.endswith('.stl'): bpy.ops.wm.stl_import(filepath=p)
    elif p.endswith('.ply'): bpy.ops.wm.ply_import(filepath=p)
    elif p.endswith('.obj'): bpy.ops.wm.obj_import(filepath=p)
objs=[o for o in bpy.context.scene.objects if o.type=='MESH']
# colour from filename for stl parts: name__RRGGBB.stl
for o in objs:
    if '__' in o.name:
        hexc=o.name.split('__')[-1][:6]
        try:
            c=[int(hexc[i:i+2],16)/255 for i in (0,2,4)]
            m=bpy.data.materials.new(o.name); m.diffuse_color=(*c,1); o.data.materials.clear(); o.data.materials.append(m)
        except ValueError: pass
    for m in ([] if '__' in o.name else o.data.materials):
        if m and m.use_nodes:
            b=m.node_tree.nodes.get('Principled BSDF')
            if b: m.diffuse_color=b.inputs['Base Color'].default_value
mn=mathutils.Vector((1e9,)*3); mx=-mn
for o in objs:
    for v in o.bound_box:
        w=o.matrix_world@mathutils.Vector(v); mn=mathutils.Vector(map(min,mn,w)); mx=mathutils.Vector(map(max,mx,w))
c=(mn+mx)/2; r=(mx-mn).length/2
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'
sh=sc.display.shading; sh.light='STUDIO'; sh.color_type='MATERIAL'; sh.show_object_outline=True; sh.show_cavity=True; sh.cavity_type='WORLD'
sc.render.resolution_x=900; sc.render.resolution_y=900; sc.render.film_transparent=False
sc.world=bpy.data.worlds.new('w'); sc.world.color=(0.9,0.9,0.9)
cam=bpy.data.cameras.new('c'); co=bpy.data.objects.new('c',cam); sc.collection.objects.link(co); sc.camera=co
cam.type='ORTHO'; cam.ortho_scale=r*2.1
import os
if os.environ.get('RZ_C'):
    c=mathutils.Vector([float(t) for t in os.environ['RZ_C'].split(',')]); cam.ortho_scale=float(os.environ.get('RZ_S','20'))
views=dict(front=(0,-1,0.15),back=(0,1,0.15),left=(-1,0,0.15),right=(1,0,0.15),q1=(-1,-1,0.7),q2=(1,1,0.7),top=(0.001,-0.001,1),under=(0.3,-0.4,-1),e2=(-0.37,0.93,0.12),e2q=(-0.8,0.6,0.3))
sel=argv[2].split(',') if len(argv)>2 else list(views)
for name in sel:
    d=mathutils.Vector(views[name]).normalized()
    co.location=c+d*r*4; co.rotation_euler=(-d).to_track_quat('-Z','Y').to_euler()
    cam.clip_end=r*10
    sc.render.filepath=f'{outp}_{name}.png'; bpy.ops.render.render(write_still=True)
