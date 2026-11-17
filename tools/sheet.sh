#!/bin/bash
# sheet.sh <dir-with-stls> <outname>  -> renders/<outname>_sheet.png (front, back, q1, q2, left)
D=$1; N=$2; R=$(dirname $0)/..
blender -b -P $R/tools/render.py -- $D $R/renders/$N front,back,q1,q2,top >/dev/null 2>&1
$R/venv/bin/python -c "
from PIL import Image
ims=[Image.open('$R/renders/${N}_'+v+'.png').convert('RGB').resize((500,500)) for v in ['front','q1','back','q2','top']]
s=Image.new('RGB',(2500,500));[s.paste(im,(i*500,0)) for i,im in enumerate(ims)];s.save('$R/renders/${N}_sheet.png')"
