# Map Recall ornaments

Four-colour, 3D-printable Christmas ornaments of Amsterdam buildings (from the Map Recall game models plus photos
and open data), each with a hanging ring. Every building comes in a solid version and a **lit** version: a hollow
1.6 mm shell with thin white window panes, open at the bottom, plus a separate **snap-on bottom**
(`out/<name>-lit/base/`) so a small battery LED can sit inside.

| building | builder |
| --- | --- |
| Het Pakhuis, Haparandaweg 902–950 (with / without lettering) | `builders/pakhuis.py` |
| Eerste Helmersstraat 148 | `builders/helmers.py` |
| Da Costakade 13 | `builders/dacosta.py` |

## Build

```sh
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
ORN_LIT=0 ./venv/bin/python builders/dacosta.py        # -> out/da-costakade-13/
ORN_LIT=1 ./venv/bin/python builders/dacosta.py        # -> out/da-costakade-13-lit/ (+ base/)
ORN_LETTERS=0 ORN_LIT=0 ./venv/bin/python builders/pakhuis.py
./venv/bin/python tools/demo_export.py                 # refresh demo/models + demo/meta.json
python3 -m http.server -d demo 8765                    # viewer
```

Each `out/<name>/` holds one STL per colour, `<name>-4colour.3mf` (parts named with their hex colour), and a
printability report. Rendering previews: `blender -b -P tools/render.py -- out/<name> <prefix> q1,front,back`.

## Snap-on bottom (`tools/basecap.py`)

A 1.6 mm plate flush with the footprint and a 5 mm plug that slides into the open bottom (0.2 mm clearance,
chamfered lead-in). A 0.3 mm bead round the plug clicks into a 0.35 mm groove cut inside the shell wall. A notch
under the longest edge takes a fingernail or coin to pop it off. Print it plate-down, no supports.

## Printability rules

45° undersides and ceilings, ≥0.5 mm features, 0.8 mm colour inlays, open railings (kick plate, top rail, posts).
