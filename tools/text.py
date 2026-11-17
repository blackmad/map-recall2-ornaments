"""Glyph outlines from a system font as manifold CrossSections (units: metres on the facade)."""
import numpy as np
from fontTools.ttLib import TTCollection, TTFont
from fontTools.pens.basePen import BasePen
from manifold3d import CrossSection, FillRule

class _Flat(BasePen):
    def __init__(self, gs, steps=8):
        super().__init__(gs); self.contours = []; self.cur = []; self.steps = steps
    def _moveTo(self, p): self.cur = [p]
    def _lineTo(self, p): self.cur.append(p)
    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, self.steps + 1)[1:]:
            a = (1 - t) ** 3; b = 3 * (1 - t) ** 2 * t; c = 3 * (1 - t) * t * t; d = t ** 3
            self.cur.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        for t in np.linspace(0, 1, self.steps + 1)[1:]:
            a = (1 - t) ** 2; b = 2 * (1 - t) * t; c = t * t
            self.cur.append((a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1]))
    def _closePath(self):
        if len(self.cur) > 2: self.contours.append(np.array(self.cur, float))
        self.cur = []
    _endPath = _closePath

def load_font(path, index=0):
    return TTCollection(path).fonts[index] if path.endswith('.ttc') else TTFont(path)

def glyph(font, ch):
    gs = font.getGlyphSet(); cmap = font.getBestCmap()
    name = cmap[ord(ch)]; pen = _Flat(gs); gs[name].draw(pen)
    return pen.contours, gs[name].width

def cap_height(font):
    os2 = font['OS/2']
    return getattr(os2, 'sCapHeight', 0) or font['head'].unitsPerEm * 0.7

def letter(font, ch, x_left, z_base, height):
    """CrossSection of one glyph, cap height `height`, left edge at x_left, baseline z_base."""
    cont, adv = glyph(font, ch)
    if not cont: return None, adv * height / cap_height(font)
    s = height / cap_height(font)
    xs = np.vstack(cont)[:, 0]; xmin = xs.min()
    cs = CrossSection([np.c_[(c[:, 0] - xmin) * s + x_left, c[:, 1] * s + z_base] for c in cont], FillRule.EvenOdd)
    return cs, (xs.max() - xmin) * s

def centred_letter(font, ch, x_centre, z_base, height):
    cs, w = letter(font, ch, 0.0, z_base, height)
    if cs is None: return None, w
    return cs.translate([x_centre - w / 2, 0]), w
